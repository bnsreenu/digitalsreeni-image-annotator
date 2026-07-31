"""``sreeni-cli`` argument parsing and dispatch (issue #76, ADR-041).

Exit codes are part of the contract, because the point of ``validate`` is to be
a CI gate:

* ``0`` — success
* ``1`` — usage error, unreadable input, or a failed operation
* ``2`` — ``validate`` found issues at or above the configured severity

Human-readable progress goes to stderr and machine-readable output to stdout,
so ``sreeni-cli validate ... | jq`` works while the narration stays visible.
"""

import argparse
import sys

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_FINDINGS = 2


def build_parser():
    parser = argparse.ArgumentParser(
        prog="sreeni-cli",
        # ASCII only in anything the CLI prints. A Windows console under a
        # legacy code page renders non-ASCII as mojibake when output is
        # redirected, which is exactly what a CI job or a batch script does.
        description=(
            "Headless operations on DigitalSreeni Image Annotator projects. "
            "Training is deliberately out of scope - use the GUI for that."
        ),
    )
    parser.add_argument(
        "--debug", action="store_true", help="verbose logging on stderr"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    export = subparsers.add_parser(
        "export", help="export a project to an annotation format"
    )
    export.add_argument("--project", required=True, help="path to the .iap file")
    export.add_argument(
        "--format", required=True,
        choices=sorted(EXPORT_FORMATS),
        help="target annotation format",
    )
    export.add_argument("--out", required=True, help="output directory")
    export.add_argument(
        "--val-split", type=int, default=0,
        help="percent of images held out for validation (YOLO formats)",
    )

    convert = subparsers.add_parser(
        "convert", help="convert between annotation formats, no project needed"
    )
    convert.add_argument("--in", dest="source", required=True,
                         help="input file or directory")
    convert.add_argument(
        "--from", dest="source_format", required=True,
        choices=sorted(IMPORT_FORMATS), help="input format",
    )
    convert.add_argument(
        "--to", dest="target_format", required=True,
        choices=sorted(EXPORT_FORMATS), help="output format",
    )
    convert.add_argument("--out", required=True, help="output directory")
    convert.add_argument(
        "--images", default=None,
        help="directory holding the images (defaults next to the input)",
    )

    validate = subparsers.add_parser(
        "validate",
        help="run the annotation QC rules; non-zero exit on findings",
    )
    validate.add_argument("--project", required=True, help="path to the .iap file")
    validate.add_argument("--json", dest="json_report", default=None,
                          help="write the findings to this JSON file")
    validate.add_argument(
        "--fail-on", default="error", choices=["error", "warning", "info", "never"],
        help="lowest severity that makes the command exit non-zero",
    )

    predict = subparsers.add_parser(
        "predict", help="run a model over a folder of images"
    )
    predict.add_argument("--model", required=True, help="path to a .pt checkpoint")
    predict.add_argument("--images", required=True, help="directory of images")
    predict.add_argument("--out", required=True, help="output directory")
    predict.add_argument(
        "--format", default="coco", choices=["coco", "yolov5"],
        help="output annotation format",
    )
    predict.add_argument(
        "--conf", type=float, default=0.25, help="confidence threshold"
    )

    # No arguments: it reports on the environment it is running in. Its value is
    # that it works when the GUI does not -- the CLI never imports Qt (ADR-041), so
    # this still runs in an environment whose Qt is broken (issue #92).
    subparsers.add_parser(
        "doctor",
        help="report the Qt/PyQt6 environment and diagnose import failures",
    )

    return parser


# Format names kept as CLI-friendly slugs and mapped to the internal labels, so
# a rename of a GUI dropdown string cannot silently break a build script.
EXPORT_FORMATS = {
    "coco": "COCO JSON",
    "yolov4": "YOLO (v4 and earlier)",
    "yolov5": "YOLO (v5+)",
    "voc": "Pascal VOC (BBox)",
    "voc-seg": "Pascal VOC (BBox + Segmentation)",
    "labeled-images": "Labeled Images",
    "semantic": "Semantic Labels",
}

IMPORT_FORMATS = {
    "coco": "COCO JSON",
    "yolov4": "YOLO (v4 and earlier)",
    "yolov5": "YOLO (v5+)",
    "voc": "Pascal VOC",
}


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    import logging

    from ..core.logging_config import configure

    configure(level=logging.DEBUG if args.debug else logging.INFO)

    # Commands are imported here, not at module load, so `--help` and a
    # `validate` run never pay for the export machinery, and `predict` is the
    # only path that ever touches torch.
    from . import commands

    handlers = {
        "export": commands.run_export,
        "convert": commands.run_convert,
        "validate": commands.run_validate,
        "predict": commands.run_predict,
        "doctor": commands.run_doctor,
    }
    try:
        return handlers[args.command](args)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover - console entry point
    sys.exit(main())
