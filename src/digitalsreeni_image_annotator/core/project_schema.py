"""Lightweight structural validation of loaded `.iap` project data (issue #42).

Pure — no Qt imports — so it is unit-testable in isolation and reusable by the
unsaved-project recovery restore path. It answers one question: "is this dict
shaped enough like a project to load without a mid-load crash?" It deliberately
does NOT reject unknown keys — the `.iap` format grows over time (keypoint
schemas, DINO config, relative paths), and a strict whitelist would break
forward compatibility.
"""


def _is_str(x):
    return isinstance(x, str)


def validate_project_data(data):
    """Return a list of human-readable problems; empty list means OK.

    Checks only load-critical shape, leniently:
      - top level is a dict
      - ``images`` is a list of dicts, each with a string ``file_name``
      - ``classes`` (if present) is a list of dicts with string ``name``/``color``
      - ``image_paths`` / ``image_paths_rel`` (if present) are dict[str, str]
      - ``notes`` (if present) is a string
    """
    problems = []

    if not isinstance(data, dict):
        return ["Project data is not a JSON object."]

    images = data.get("images")
    if not isinstance(images, list):
        problems.append("'images' must be a list.")
    else:
        for i, img in enumerate(images):
            if not isinstance(img, dict):
                problems.append(f"images[{i}] must be an object.")
            elif not _is_str(img.get("file_name")):
                problems.append(f"images[{i}] is missing a string 'file_name'.")

    classes = data.get("classes")
    if classes is not None:
        if not isinstance(classes, list):
            problems.append("'classes' must be a list.")
        else:
            for i, cls in enumerate(classes):
                if not isinstance(cls, dict):
                    problems.append(f"classes[{i}] must be an object.")
                else:
                    if not _is_str(cls.get("name")):
                        problems.append(f"classes[{i}] is missing a string 'name'.")
                    if not _is_str(cls.get("color")):
                        problems.append(f"classes[{i}] is missing a string 'color'.")

    for key in ("image_paths", "image_paths_rel"):
        val = data.get(key)
        if val is not None:
            if not isinstance(val, dict) or not all(
                _is_str(k) and _is_str(v) for k, v in val.items()
            ):
                problems.append(f"'{key}' must be an object of string→string.")

    notes = data.get("notes")
    if notes is not None and not _is_str(notes):
        problems.append("'notes' must be a string.")

    return problems
