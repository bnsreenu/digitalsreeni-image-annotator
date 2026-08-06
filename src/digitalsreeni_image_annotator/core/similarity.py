"""Near-duplicate clustering over image embeddings (issue #72).

Datasets collected from video, microscopy time series or burst captures are
full of near-identical frames. Annotating forty near-duplicates costs forty
times the effort and teaches the model roughly what five would. Conversely a
dataset can be missing whole regions of appearance space without anyone
noticing. The app had no notion of image similarity, so neither the redundancy
nor the gaps were visible.

Qt-free and model-free: everything here operates on plain float vectors. The
vectors come from wherever — CLIP today, DINOv2 tomorrow — and this module does
not care, which is what lets the backend be swapped and compared without
touching the clustering.

**Threshold-based connected components, not k-means.** The cluster count is not
known in advance and the user should not have to guess it; worse, k-means would
return a partition of *every* image whether or not any of them are actually
similar. Connected components above a similarity threshold returns exactly the
groups that are genuinely close, and nothing else — and re-clustering at a new
threshold is instant, since the embeddings do not change.
"""

import collections
import math

import numpy as np

DEFAULT_SIMILARITY = 0.95

# Appearance modes are components at a deliberately *low* threshold: the
# question is "how many distinct kinds of image are in here", which is coarser
# than "which of these are near-duplicates". The value is a heuristic and it is
# model-dependent -- CLIP and DINOv2 do not put the same numbers on the same
# pair -- so every place that reports a mode count also reports the threshold
# it used. A number stated without its threshold would read as ground truth.
MODE_SIMILARITY = 0.80

# One row block of the similarity matrix is capped at this many elements, so
# peak memory for the pairwise work is a fixed ~16 MB (float32) instead of
# growing with n^2. The block *row count* is derived from this and n, so a
# larger dataset gets narrower blocks rather than a bigger allocation.
#
# Every pairwise routine here has to use it. `representative` did not, and a
# single video clusters into ONE component -- so "a cluster" was routinely the
# whole dataset and the unblocked k x k product was a 1.6 GB allocation at the
# limit below.
_MAX_BLOCK_ELEMENTS = 4_000_000

# Above this many images the run stops being worth starting. The old limit was
# 3000 and it measured the implementation rather than the problem: a pure-
# Python double loop recomputing both vector norms per pair. Vectorised, 20 000
# images re-cluster in a few seconds with bounded peak memory, and the binding
# cost has moved to *embedding* -- one forward pass per image through CLIP or
# DINOv2, which nothing here makes cheaper.
#
# Measured on this module at 20 000 images with 768-d vectors: the (n, d)
# matrix is a 61 MB floor that every routine pays, and the blocked pairwise
# work sits on top of it -- analyse 99 MB, representative 94 MB, cohesion
# 113 MB. The floor is inherent; the pairwise term is what blocking bounds.
#
# So the limit is now set by what a person will wait for, not by what fits in
# RAM, and the message says which.
ALL_PAIRS_LIMIT = 20_000

ALL_PAIRS_LIMIT_MESSAGE = (
    f"This project has more than {ALL_PAIRS_LIMIT} images. Every one of them "
    "has to go through the embedding model once before anything can be "
    "compared, and at this scale that alone runs for a long time -- hours "
    "without a GPU. Comparing them afterwards is quick by contrast.\n\n"
    "Narrow the selection (for example by group) and try again."
)


def l2_normalise(vector):
    """Unit-length copy of ``vector``, so cosine similarity is a plain dot
    product. A zero vector is returned unchanged rather than raising — a blank
    image is a legitimate input."""
    norm = math.sqrt(sum(float(v) * float(v) for v in vector))
    if norm == 0:
        return [float(v) for v in vector]
    return [float(v) / norm for v in vector]


def cosine_similarity(a, b):
    """Dot product of two vectors, normalising first if they are not already.

    Normalising defensively costs little and makes the function correct for
    callers that hand in raw model output.
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    norm_a = math.sqrt(sum(float(x) * float(x) for x in a))
    norm_b = math.sqrt(sum(float(y) * float(y) for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _stack(names, embeddings):
    """An ``(len(names), d)`` float32 matrix of unit rows, in ``names`` order.

    Every cosine similarity below is then a plain dot product, and a whole
    block of them is one matrix product -- which is the entire point: the
    pure-Python version recomputed both vector norms for every pair.

    float32, not float64: the vectors come out of the model as float32, so
    widening them stores four bytes of precision the number never had.

    Two inputs are handled rather than rejected, because both used to be:

    * a **zero vector** (a blank image) normalises to itself, so its dot
      product with everything is 0 -- it resembles nothing, which is correct;
    * a vector of the **wrong length** gets a zero row. Mixed dimensions can
      only happen if two models' outputs are mixed, which the cache prevents
      by keying on model identity -- but the old ``cosine_similarity`` returned
      0.0 for a length mismatch instead of raising, and silently crashing a
      curation run on a corrupt cache entry would be a worse trade. The
      reference length is the most common one present.
    """
    # `np.asarray`, not a truthiness test: a caller may hold its embeddings as
    # NumPy arrays already (the curation controller does, so that 20 000 of
    # them cost 60 MB rather than the 500 MB the equivalent Python float lists
    # would), and `array or []` raises rather than doing anything sensible.
    vectors = [
        np.asarray(embeddings.get(name, ()), dtype=np.float32).ravel()
        for name in names
    ]
    lengths = collections.Counter(
        vector.size for vector in vectors if vector.size
    )
    reference = lengths.most_common(1)[0][0] if lengths else 0

    matrix = np.zeros((len(names), reference), dtype=np.float32)
    for row, vector in enumerate(vectors):
        if vector.size == reference:
            matrix[row] = vector
    # einsum, not `(matrix * matrix).sum(axis=1)`: the latter materialises a
    # second full copy of the matrix purely to sum it away, which doubled the
    # peak of every routine downstream. Every pairwise pass here is carefully
    # blocked and then all of them call this, so an unblocked allocation here
    # sets the real ceiling.
    norms = np.sqrt(np.einsum("ij,ij->i", matrix, matrix))
    norms[norms == 0] = 1.0
    matrix /= norms[:, None]
    return matrix


def _matrix(embeddings):
    """``(sorted_names, unit_matrix)`` for an embedding mapping."""
    names = sorted(embeddings or {})
    return names, _stack(names, embeddings or {})


def _block_rows(n):
    """How many rows to multiply at once so one block stays inside the budget.

    At least one row, so a very wide embedding can never divide to zero and
    loop forever.
    """
    return max(1, _MAX_BLOCK_ELEMENTS // max(1, n))


class _Labelling:
    """Connected components under construction, one label per row.

    ``labels[i]`` is the smallest row index in ``i``'s component so far.
    Merging two components rewrites one label across the whole array, which is
    O(n) -- but two components can only merge n-1 times in total, so the
    bookkeeping over a whole scan is O(n^2) elementwise, done in NumPy, and it
    never allocates an edge list.

    That last property is the one that matters. Union-find over an explicit
    edge list computes the same answer in O(edges), and a project of 20 000
    near-identical frames -- precisely the case someone opens this tool to
    diagnose -- has 200 million of them.
    """

    def __init__(self, n):
        self.labels = np.arange(n)
        # Once everything is one component there is nothing left to merge, and
        # the per-row work below can be skipped entirely for the rest of the
        # scan. This is the fully-redundant case, so it is worth the counter.
        self.distinct = n

    def absorb(self, hits, counts, start):
        """Merge each row of a block with the rows it is adjacent to.

        ``hits`` is the block's boolean adjacency, ``counts`` its per-row
        neighbour count -- computed once for the block, so a row with no
        neighbours (the common case at a high threshold) costs nothing beyond
        that.
        """
        if self.distinct <= 1:
            return
        labels = self.labels
        for local in np.nonzero(counts)[0]:
            neighbours = labels[hits[local]]
            own = labels[start + local]
            low, high = neighbours.min(), neighbours.max()
            if low == high == own:
                # Already one component: the overwhelmingly common case once a
                # burst of frames has merged, and it costs no allocation.
                continue
            stale = np.unique(np.append(neighbours, own))
            target = stale[0]
            for label in stale[1:]:
                labels[labels == label] = target
                self.distinct -= 1
            if self.distinct <= 1:
                return

    def groups(self):
        """Row indices per component, each ascending."""
        components = {}
        for index, label in enumerate(self.labels.tolist()):
            components.setdefault(label, []).append(index)
        return list(components.values())


def _scan(matrix, thresholds):
    """One blocked pass over every pair. ``(labellings, nearest)``.

    A single matrix product per block serves *all* the questions asked of it:
    each threshold in ``thresholds`` gets its own component labelling, and
    ``nearest`` -- each row's highest similarity to any other row -- comes off
    the same product. The report needs near-duplicate clusters, isolated
    images and coarse appearance modes; computing them separately would sweep
    the same pairs three times.

    Blocked rather than one big product, with the block height derived from n
    (:func:`_block_rows`), so peak memory is bounded by a constant instead of
    the n^2 the old implementation would have needed had it materialised
    anything at all.
    """
    n = matrix.shape[0]
    labellings = [_Labelling(n) for _ in thresholds]
    nearest = np.full(n, -np.inf, dtype=np.float32)
    chunk = _block_rows(n)

    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        similarities = matrix[start:stop] @ matrix.T
        # A row is not its own neighbour, and must not be its own nearest.
        similarities[np.arange(stop - start), np.arange(start, stop)] = -np.inf
        nearest[start:stop] = similarities.max(axis=1)
        for labelling, threshold in zip(labellings, thresholds):
            hits = similarities >= threshold
            labelling.absorb(hits, hits.sum(axis=1), start)

    return labellings, nearest


def _named_groups(names, labelling):
    """Components as name lists, largest first then alphabetically."""
    groups = [[names[index] for index in group] for group in labelling.groups()]
    groups.sort(key=lambda group: (-len(group), group[0]))
    return groups


def _pairwise_stats(matrix):
    """``(minimum, mean)`` over the distinct pairs of ``matrix``' rows."""
    n = matrix.shape[0]
    chunk = _block_rows(n)
    minimum, total, count = math.inf, 0.0, 0
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        similarities = matrix[start:stop] @ matrix.T
        # Upper triangle only, so each pair is counted once and no row is
        # compared with itself.
        upper = similarities[
            np.arange(start, stop)[:, None] < np.arange(n)[None, :]
        ]
        if upper.size:
            minimum = min(minimum, float(upper.min()))
            total += float(upper.sum())
            count += upper.size
    if not count:
        return None
    return minimum, total / count


def cluster(embeddings, threshold=DEFAULT_SIMILARITY):
    """Group names whose embeddings are mutually reachable above ``threshold``.

    ``embeddings`` is ``{name: vector}``. Returns a list of clusters, each a
    list of names sorted alphabetically, with clusters ordered largest first.
    **Singletons are omitted** — an image similar to nothing is not a
    near-duplicate cluster, and including every one of them would bury the
    handful of findings that matter.

    Connected components, so A~B and B~C puts all three together even if A and
    C are below the threshold. That is the right transitive behaviour for a
    burst of frames drifting slowly: consecutive pairs are near-identical while
    the ends of the run are not.
    """
    names, matrix = _matrix(embeddings)
    if len(names) < 2:
        return []
    labellings, _nearest = _scan(matrix, [threshold])
    return [
        group for group in _named_groups(names, labellings[0]) if len(group) > 1
    ]


def modes(embeddings, threshold=MODE_SIMILARITY):
    """A **partition** of every name into coarse appearance modes.

    Same machinery as :func:`cluster` at a lower threshold and with singletons
    kept, and the difference between the two is the whole point. ``cluster``
    answers "what is redundant here" and drops anything that resembles nothing;
    ``modes`` answers "what kinds of image are in here at all", so an image
    resembling nothing is not noise to be dropped -- it is a mode of one, and
    a dataset made of forty of those is the interesting finding.

    Ordered largest first, then alphabetically, like ``cluster``.
    """
    names, matrix = _matrix(embeddings)
    if not names:
        return []
    labellings, _nearest = _scan(matrix, [threshold])
    return _named_groups(names, labellings[0])


def analyse(
    embeddings, threshold=DEFAULT_SIMILARITY, mode_threshold=MODE_SIMILARITY
):
    """``{"clusters", "outliers", "modes"}`` from a single pass over the pairs.

    The report wants all three at once, and asking for them one at a time
    sweeps the same pairs three times. This is what the dialog calls, so moving
    the threshold slider costs one pass rather than three -- the difference
    between a slider that tracks and one that stutters on a large project.

    ``modes`` is computed at its own, lower threshold in the same pass: two
    comparisons against one matrix product cost nothing next to the product.
    It is **clamped** to no higher than ``threshold``, and the effective value
    is returned: the slider reaches 0.50, well below the 0.80 default, and
    modes finer than the near-duplicate clusters they are supposed to
    generalise would invert the relationship the report describes.
    """
    mode_threshold = min(mode_threshold, threshold)
    names, matrix = _matrix(embeddings)
    if len(names) < 2:
        return {
            "clusters": [],
            "outliers": [],
            "modes": [[name] for name in names],
            "mode_threshold": mode_threshold,
        }

    labellings, nearest = _scan(matrix, [threshold, mode_threshold])
    return {
        "clusters": [
            group for group in _named_groups(names, labellings[0]) if len(group) > 1
        ],
        "outliers": [
            name for index, name in enumerate(names) if nearest[index] < threshold
        ],
        "modes": _named_groups(names, labellings[1]),
        "mode_threshold": mode_threshold,
    }


def cohesion(cluster_names, embeddings):
    """``{"min", "mean"}`` pairwise similarity inside a cluster, or ``None``.

    Makes the one known weakness of connected components *visible* instead of
    arguing about it. Transitivity is the right call for a slow pan -- each
    consecutive pair is near-identical while the ends of the run are not -- but
    it does mean a cluster can be a chain rather than a blob, and the report
    otherwise presents both the same way.

    A compact cluster has ``min`` close to ``mean``. A chained one has a ``min``
    well below it, and that is the cluster whose "suggested representative"
    deserves a second look before anything is skipped on its account.

    ``None`` for fewer than two names: a single image has no pairs, and
    reporting 1.0 for it would be inventing a measurement.
    """
    present = [name for name in (cluster_names or []) if name in (embeddings or {})]
    if len(present) < 2:
        return None
    stats = _pairwise_stats(_stack(present, embeddings))
    if stats is None:
        return None
    minimum, mean = stats
    return {"min": minimum, "mean": mean}


def representative(cluster_names, embeddings):
    """The most central member of a cluster — the one to keep.

    Chosen as the highest mean similarity to the rest, i.e. the medoid. Picking
    the first alphabetically would be arbitrary; the medoid is the frame that
    best stands in for the group.
    """
    if not cluster_names:
        return None
    if len(cluster_names) == 1:
        return cluster_names[0]
    matrix = _stack(cluster_names, embeddings)
    count = matrix.shape[0]
    totals = np.empty(count, dtype=np.float32)
    # Blocked like everything else here. A single video clusters into ONE
    # component, so `cluster_names` can be the whole dataset -- an unblocked
    # k x k product would be a 1.6 GB allocation at the supported ceiling, and
    # the dialog calls this once per cluster.
    chunk = _block_rows(count)
    for start in range(0, count, chunk):
        stop = min(start + chunk, count)
        similarities = matrix[start:stop] @ matrix.T
        # Mean similarity to the *others*: the row sum less the row's
        # similarity to itself, which is 1 for any real vector and 0 for a zero
        # one -- so it is subtracted rather than assumed. The divisor is the
        # same for every row, so argmax over the sums is the same answer.
        own = similarities[np.arange(stop - start), np.arange(start, stop)]
        totals[start:stop] = similarities.sum(axis=1) - own
    # argmax takes the first of a tie, matching the strict `>` the loop used.
    return cluster_names[int(np.argmax(totals))]


def outliers(embeddings, threshold=DEFAULT_SIMILARITY):
    """Names whose nearest neighbour is below ``threshold``.

    The other half of the diversity picture: these are the images nothing else
    in the dataset resembles, which is where coverage is thinnest.
    """
    names, matrix = _matrix(embeddings)
    if len(names) < 2:
        return []
    _labellings, nearest = _scan(matrix, [])
    return [name for index, name in enumerate(names) if nearest[index] < threshold]


def summarise(clusters, total_images):
    """Headline numbers for the report.

    ``redundant`` is what could be *skipped*: every cluster member beyond its
    representative. That is the number that turns a finding into a decision.
    """
    clustered = sum(len(group) for group in clusters)
    redundant = sum(len(group) - 1 for group in clusters)
    return {
        "clusters": len(clusters),
        "clustered_images": clustered,
        "redundant_images": redundant,
        "total_images": total_images,
        "largest_cluster": max((len(g) for g in clusters), default=0),
    }
