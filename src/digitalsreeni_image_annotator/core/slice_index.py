"""Name-keyed lookup across every loaded slice collection.

Every exporter needs the same thing: given an annotation key, is it a slice or
a frame, and if so what are its pixels? They each answered it by building
``{name: qimage for name, qimage in slices}`` and then rescanning
``image_slices`` linearly on a miss.

Both halves of that are wrong on a lazy collection. The comprehension drives
``LazySliceList.__iter__``, which materialises **every** slice and holds them
all at once -- exactly what the bounded LRU of issue #45 exists to prevent, and
an out-of-memory risk rather than a slowdown on a 2560-slice 5D stack. The
rescan then decodes the whole of every other collection per miss.

So: index by **name** (free -- no decode), and resolve one slice at a time
through the collection's own LRU-backed ``get``.

Qt-free (ADR-041): it only ever touches ``.names`` and ``.get``, never a
``QImage``, so ``io.export_formats`` can import it at module level and the
headless CLI stays importable without a display.
"""


def slice_index(slices, image_slices):
    """``{slice_name: collection}`` over every known slice, decoding nothing.

    ``slices`` is the active collection, ``image_slices`` the per-stack mapping;
    a video's frames live in the latter exactly like a TIFF stack's slices
    (issues #45/#47), which is what lets an annotated video export at all.

    The active collection is indexed first and wins ties, matching the previous
    behaviour where its map was consulted before the per-stack fallback.
    """
    index = {}
    for collection in (slices, *(image_slices or {}).values()):
        if not collection:
            continue
        # `.names` on a LazySliceList; a plain [(name, qimage), ...] otherwise
        # (legacy call sites and several tests still hand in the latter).
        names = getattr(collection, "names", None)
        if names is None:
            names = [name for name, _ in collection]
        for name in names:
            index.setdefault(name, collection)
    return index


def resolve_slice_image(index, name):
    """The QImage for a slice name, or ``None``. At most one decode."""
    collection = index.get(name)
    if collection is None:
        return None
    getter = getattr(collection, "get", None)
    if getter is not None:
        return getter(name)
    return next((image for n, image in collection if n == name), None)
