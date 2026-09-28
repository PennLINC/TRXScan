"""Generate a pooch registry (``<file> <sha256>`` per line) for a release directory::

    python -m trxscan.data.checksums <dir> > trxscan/data/registry_<bundle>.txt
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    d = Path(args[0])
    for p in sorted(x for x in d.iterdir() if x.is_file()):
        print(f"{p.name} {sha256(p)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
