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

import math

DEFAULT_SIMILARITY = 0.95

# Above this many images an all-pairs comparison stops being reasonable
# (n^2/2 similarities). The caller is told rather than left to discover it as
# an unresponsive dialog.
ALL_PAIRS_LIMIT = 3000

ALL_PAIRS_LIMIT_MESSAGE = (
    f"This project has more than {ALL_PAIRS_LIMIT} images. Similarity analysis "
    "compares every pair, so memory and time grow with the square of the count — "
    "running it here would exhaust RAM rather than finish slowly. Narrow the "
    "selection (for example by group) and try again."
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
    names = sorted(embeddings or {})
    parent = {name: name for name in names}

    def find(name):
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    def union(a, b):
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if cosine_similarity(embeddings[a], embeddings[b]) >= threshold:
                union(a, b)

    groups = {}
    for name in names:
        groups.setdefault(find(name), []).append(name)

    clusters = [sorted(group) for group in groups.values() if len(group) > 1]
    clusters.sort(key=lambda group: (-len(group), group[0]))
    return clusters


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
    best, best_score = None, -2.0
    for name in cluster_names:
        others = [other for other in cluster_names if other != name]
        score = sum(
            cosine_similarity(embeddings[name], embeddings[other]) for other in others
        ) / len(others)
        if score > best_score:
            best, best_score = name, score
    return best


def outliers(embeddings, threshold=DEFAULT_SIMILARITY):
    """Names whose nearest neighbour is below ``threshold``.

    The other half of the diversity picture: these are the images nothing else
    in the dataset resembles, which is where coverage is thinnest.
    """
    names = sorted(embeddings or {})
    if len(names) < 2:
        return []
    result = []
    for name in names:
        nearest = max(
            cosine_similarity(embeddings[name], embeddings[other])
            for other in names
            if other != name
        )
        if nearest < threshold:
            result.append(name)
    return result


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
