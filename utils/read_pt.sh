#!/bin/bash
file="$1"
if [[ ! -f "$file" ]]; then
  echo "File not found!"
  exit 1
fi

python - <<END
import torch
f = torch.load("$file", map_location='cpu')
if isinstance(f, dict):
    print("Keys:", list(f.keys()))
elif isinstance(f, list):
    print("Length:", len(f))
    print("Head:", f[:10])
else:
    print("Type:", type(f))
    print("Repr:", repr(f))
END