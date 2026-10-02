"""Recover collector permissions after a hard exit, without running any models.

Run only after all writers have stopped, with access to the original output tree.
Uses the same secret/symlink filtering as the normal submission finalizer.
"""
import argparse
from pathlib import Path

from firstproof_entrypoint import _finalize_output_permissions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if not args.output.is_dir() or args.output.is_symlink():
        parser.error("output must be an existing, non-symlink directory")
    warnings = _finalize_output_permissions(args.output)
    for warning in warnings:
        print(warning)
    return 1 if warnings else 0


if __name__ == "__main__":
    raise SystemExit(main())
