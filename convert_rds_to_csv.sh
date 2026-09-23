#!/usr/bin/env bash
set -euo pipefail

echo "[convert] starting RDS conversion..."
trap 'echo "[convert] failed with exit code $?" >&2' ERR

input_file="${1:?Usage: $0 INPUT.rds [OUTPUT.csv]}"
output_file="${2:-${input_file%.rds}.csv}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -x "$script_dir/.venv/Scripts/python.exe" ]]; then
    python_command="$script_dir/.venv/Scripts/python.exe"
elif command -v python >/dev/null 2>&1; then
    python_command="python"
elif command -v python3 >/dev/null 2>&1; then
    python_command="python3"
elif command -v python.exe >/dev/null 2>&1; then
    python_command="python.exe"
else
    echo "Python was not found. Install Python and pyreadr first." >&2
    exit 1
fi

echo "[convert] using Python: $python_command"

"$python_command" -u - "$input_file" "$output_file" <<'PY' &
import sys
import time
from pathlib import Path

input_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])

def status(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)

if input_path.suffix.lower() != ".rds":
    raise SystemExit(f"Expected an .rds file, got: {input_path}")
if not input_path.is_file():
    raise SystemExit(f"Input file not found: {input_path}")

status(f"Input: {input_path} ({input_path.stat().st_size / 1024 / 1024:.1f} MiB)")
status("Trying pyreadr...")

try:
    import pyreadr
    objects = pyreadr.read_r(str(input_path))
    if not objects:
        raise ValueError("no objects returned")
    data = next(iter(objects.values()))
    status("pyreadr loaded the file")
except Exception as pyreadr_error:
    status(f"pyreadr could not read the file: {pyreadr_error}")
    status("Trying rdata; this can take a few minutes...")
    try:
        import rdata
        parsed = rdata.parser.parse_file(str(input_path))
        data = rdata.conversion.convert(parsed)
        status("rdata loaded the file")
    except Exception as rdata_error:
        raise SystemExit(
            f"Could not read {input_path} with pyreadr or rdata. "
            f"pyreadr: {pyreadr_error}; rdata: {rdata_error}"
        ) from rdata_error

if not hasattr(data, "to_csv"):
    raise SystemExit(f"The RDS object is not a tabular dataframe: {type(data).__name__}")
status(f"Writing {output_path}...")
data.to_csv(output_path, index=False)
status(f"Converted {len(data):,} rows and {len(data.columns):,} columns")
status(f"Wrote: {output_path}")
PY

python_pid=$!
started_at=$SECONDS
while kill -0 "$python_pid" 2>/dev/null; do
    elapsed=$((SECONDS - started_at))
    echo "[convert] still working (${elapsed}s elapsed)..."
    sleep 10
done
wait "$python_pid"
echo "[convert] finished."