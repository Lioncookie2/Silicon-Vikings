#!/usr/bin/env python3
"""Bygg submission.zip for NorgesGruppen: run.py (og evt. utils.py + vekter) i roten av zip."""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

# Tillatte vekttyper i konkurransen
WEIGHT_SUFFIXES = frozenset({".pt", ".pth", ".onnx", ".safetensors", ".npy"})


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def submission_dir(root: Path) -> Path:
    return root / "norgesgruppen" / "submission"


def list_files_to_pack(sub: Path) -> list[Path]:
    run_py = sub / "run.py"
    if not run_py.is_file():
        raise SystemExit(f"Mangler {run_py}")

    extra_py = [
        p
        for p in sub.glob("*.py")
        if p.is_file() and p.name not in {"run.py", "utils.py"}
    ]
    if extra_py:
        names = ", ".join(sorted(p.name for p in extra_py))
        raise SystemExit(
            f"For mange/uventede .py-filer i {sub} (maks tillatt i zip: run.py + utils.py). "
            f"Fjern: {names}"
        )

    weights = sorted(
        p
        for p in sub.iterdir()
        if p.is_file() and p.suffix.lower() in WEIGHT_SUFFIXES
    )
    if not weights:
        raise SystemExit(
            "Mangler vektfil i norgesgruppen/submission/. Kopier f.eks.:\n"
            "  cp runs/ngd_yolo/baseline/weights/best.pt norgesgruppen/submission/best.pt\n"
            "eller legg inn model.onnx (run.py velger ONNX først hvis den finnes)."
        )
    if len(weights) > 3:
        raise SystemExit(
            f"For mange vektfiler (maks 3 i konkurransen): {[w.name for w in weights]}"
        )

    out: list[Path] = [run_py]
    utils_py = sub / "utils.py"
    if utils_py.is_file():
        out.append(utils_py)
    out.extend(weights)

    py_count = sum(1 for p in out if p.suffix.lower() == ".py")
    if py_count > 10:
        raise SystemExit(f"For mange Python-filer i zip ({py_count}, maks 10).")

    if len(out) > 1000:
        raise SystemExit(f"For mange filer ({len(out)}, maks 1000).")

    return out


def write_zip(paths: list[Path], out_zip: Path) -> None:
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in paths:
            # Arkivnavn = kun filnavn (run.py i rot, ikke submission/run.py)
            arcname = path.name
            zf.write(path, arcname)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pakker norgesgruppen/submission/ til submission.zip")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Utdata-zip (standard: <repo>/submission.zip)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = repo_root()
    sub = submission_dir(root)
    out_zip = args.output if args.output is not None else root / "submission.zip"

    if not sub.is_dir():
        raise SystemExit(f"Fant ikke submission-mappe: {sub}")

    paths = list_files_to_pack(sub)
    write_zip(paths, out_zip)

    print(f"Opprettet {out_zip}")
    print("Innhold:")
    for p in paths:
        print(f"  - {p.name} ({p.stat().st_size // 1024} KB)")
    print()
    print("Sjekk at run.py ligger i roten av zip:")
    print(f"  unzip -l {out_zip} | head")


if __name__ == "__main__":
    main()
