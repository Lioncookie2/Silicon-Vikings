from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def fail(message: str) -> None:
    raise RuntimeError(message)


def main() -> None:
    project_root = repo_root()
    sub_dir = project_root / "norgesgruppen" / "submission"
    run_py = sub_dir / "run.py"
    input_dir = project_root / "data" / "yolo" / "images" / "val"
    output_json = sub_dir / "test_predictions.json"

    if not run_py.exists():
        fail(f"Fant ikke submission-script: {run_py}")
    if not (sub_dir / "best.pt").exists() and not (sub_dir / "model.onnx").exists():
        fail(
            "Mangler vekt i norgesgruppen/submission/ (best.pt eller model.onnx). "
            "Kopier etter trening, deretter kjør smoke test på nytt."
        )
    if not input_dir.exists() or not input_dir.is_dir():
        fail(f"Fant ikke input-mappe: {input_dir}")

    command = [
        sys.executable,
        str(run_py),
        "--input",
        str(input_dir),
        "--output",
        str(output_json),
    ]
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        fail(
            "Kjoring av norgesgruppen/submission/run.py feilet.\n"
            f"Exit code: {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

    if not output_json.exists():
        fail(f"Outputfil ble ikke laget: {output_json}")

    with output_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        fail("Output JSON er ikke en liste.")
    if len(data) == 0:
        fail("Output JSON er tom (0 predictions).")

    print(f"Predictions: {len(data)}")
    print("Forste 3 predictions:")
    for item in data[:3]:
        print(item)


if __name__ == "__main__":
    main()
