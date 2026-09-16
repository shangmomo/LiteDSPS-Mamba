from __future__ import annotations

import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEM = "LiteDSPS_Mamba_R5_DirShare_Deterministic_Final_20260916.zip"
EXPECTED_SHA256 = "36df017861e61bbb3600591227ca6b6ac04dd72d90d426b42b05f15b05096dbb"

parts = [HERE / f"{STEM}.part{i:02d}" for i in range(1, 8)]
missing = [p.name for p in parts if not p.exists()]
if missing:
    raise FileNotFoundError(f"Missing archive parts: {missing}")

payload = b"".join(p.read_bytes() for p in parts)
digest = hashlib.sha256(payload).hexdigest()
if digest != EXPECTED_SHA256:
    raise RuntimeError(f"SHA-256 mismatch: expected {EXPECTED_SHA256}, got {digest}")

out = HERE / STEM
out.write_bytes(payload)
print(f"Wrote {out} ({len(payload)} bytes), SHA-256={digest}")
