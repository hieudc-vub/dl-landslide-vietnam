"""Command-line interface; --help does not import the GPU runtime."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def dispatch():
    parser = argparse.ArgumentParser(description="20-channel landslide benchmark")
    parser.add_argument("--debug", action="store_true", help="Show the full error traceback")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check-env", help="Check Python, dependencies and GPU")
    p = sub.add_parser("validate-data", help="Decode and hash every STACK/LABEL pair")
    p.add_argument("--data-root", required=True)
    p.add_argument("--out", required=True)
    for command in ["train", "calibrate", "evaluate", "predict", "benchmark", "run", "run-all"]:
        p = sub.add_parser(command)
        p.add_argument("--out", required=True)
        p.add_argument("--device", default="auto")
        if command in {"run", "run-all"}:
            p.add_argument("--skip-benchmark", action="store_true", help="Skip optional timing/operator profiling")
        if command in {"train", "run"}:
            p.add_argument("--config", required=True)
        if command == "run-all":
            p.add_argument("--configs", default="configs/models")
        if command in {"train", "calibrate", "evaluate", "run", "run-all"}:
            p.add_argument("--data-root", required=True)
        if command in {"calibrate", "evaluate", "predict", "benchmark"}:
            p.add_argument("--checkpoint", required=True)
        if command in {"evaluate", "predict"}:
            p.add_argument("--threshold", required=True, help="VAL threshold.json from calibrate")
        if command == "predict":
            p.add_argument("--input", required=True, help="One packed TIFF or a directory of TIFFs")
    p = sub.add_parser("summarize")
    p.add_argument("--runs", required=True)
    p.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.command == "check-env":
        from .diagnostics import check_environment
        check_environment()
        return
    if args.command == "run-all":
        configs = sorted(Path(args.configs).glob("*.json"))
        if len(configs) != 13:
            parser.error("Expected exactly 13 model configurations")
        # Separate processes release GPU memory and restart all random state per model.
        for config in configs:
            subprocess.run([sys.executable, "-m", "landslide", "run", "--config", str(config),
                "--data-root", args.data_root, "--out", str(Path(args.out) / config.stem),
                "--device", args.device] + (["--skip-benchmark"] if args.skip_benchmark else []), check=True)
        return
    if args.command == "summarize":
        from .reporting import summarize
        summarize(args.runs, args.out)
        return
    if args.command == "validate-data":
        from .data import validate_dataset
        print(validate_dataset(args.data_root, args.out))
        return
    from . import engine
    from .config import load
    device = engine.device_for(args.device)
    if args.command in {"train", "run"}:
        cfg = load(args.config)
        engine.train(cfg, args.data_root, args.out, device)
        if args.command == "run":
            checkpoint = Path(args.out) / "best_model.pth"
            calibration = Path(args.out) / "calibration"
            engine.calibrate(checkpoint, args.data_root, calibration, device)
            engine.evaluate(checkpoint, calibration / "threshold.json", args.data_root,
                            Path(args.out) / "evaluation", device)
            if not args.skip_benchmark:
                engine.benchmark(checkpoint, Path(args.out) / "benchmark.json", device)
    elif args.command == "calibrate":
        engine.calibrate(args.checkpoint, args.data_root, args.out, device)
    elif args.command == "evaluate":
        engine.evaluate(args.checkpoint, args.threshold, args.data_root, args.out, device)
    elif args.command == "predict":
        engine.predict(args.checkpoint, args.threshold, args.input, args.out, device)
    elif args.command == "benchmark":
        engine.benchmark(args.checkpoint, args.out, device)


def main():
    """Keep failures readable, preserve a detailed local log for troubleshooting."""
    try:
        dispatch()
    except KeyboardInterrupt:
        print("\nStopped by the user. Completed artifacts remain in the output folder.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        if "--debug" in sys.argv:
            raise
        from datetime import datetime
        import traceback
        messages = {
            FileNotFoundError: "A required file or folder was not found. Check the path and extract the dataset first.",
            FileExistsError: "The output folder already exists. Choose a new output folder; existing results are preserved.",
            ModuleNotFoundError: "A required Python package is missing. Run the Windows setup script with the project environment.",
            PermissionError: "Windows denied file access. Choose a writable project/output folder and close programs locking the file.",
        }
        hint = next((v for k, v in messages.items() if isinstance(error, k)),
                    "The operation could not finish. See docs/troubleshooting.md and the local error log.")
        if "out of memory" in str(error).lower():
            hint = "GPU memory is insufficient. Close other GPU programs. A smaller batch requires a separate experimental config and is not the fixed benchmark."
        print(f"ERROR: {hint}", file=sys.stderr)
        print(f"Details: {str(error)[:500]}", file=sys.stderr)
        try:
            log = Path("logs") / ("error_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".txt")
            log.parent.mkdir(exist_ok=True)
            log.write_text(traceback.format_exc(), encoding="utf-8")
            print(f"Detailed log: {log}. Review it before sharing; it can contain local paths.", file=sys.stderr)
        except OSError:
            print("Could not write a log. Add --debug before the command to show the traceback.", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
