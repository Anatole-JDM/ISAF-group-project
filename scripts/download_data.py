"""Fetch the Nashville stops file reproducibly (120 MB zipped)."""
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402


def main() -> None:
    C.DATA_RAW.mkdir(parents=True, exist_ok=True)
    if C.STOPS_ZIP.exists():
        print(f"already present: {C.STOPS_ZIP} ({C.STOPS_ZIP.stat().st_size/2**20:.0f} MB)")
        return
    print(f"downloading {C.STOPS_URL}")
    with requests.get(C.STOPS_URL, stream=True, timeout=120) as r:
        r.raise_for_status()
        done = 0
        with open(C.STOPS_ZIP, "wb") as fh:
            for block in r.iter_content(1 << 20):
                fh.write(block)
                done += len(block)
                print(f"\r  {done/2**20:.0f} MB", end="", flush=True)
    print(f"\nsaved to {C.STOPS_ZIP}")


if __name__ == "__main__":
    main()
