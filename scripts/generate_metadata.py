import os
import sys
import json
from datetime import datetime

png_root = sys.argv[1]  # z.B. pngs/gewitter
run = sys.argv[2]
date = sys.argv[3] if len(sys.argv) > 3 else datetime.utcnow().strftime("%Y%m%d")

metadata = {
    "run": run,
    "date": date,
    "generated_at": datetime.utcnow().isoformat() + "Z",
    "var_type": "gewitter",
    "timesteps": []
}

# PNG-Dateien finden
files = sorted(f for f in os.listdir(png_root) if f.endswith(".png"))

for f in files:
    # gewitter_20251008_0700.png → 20251008_0700
    name = f.replace(".png", "")
    parts = name.split("_")

    if len(parts) >= 3:
        timestep = parts[-2] + "_" + parts[-1]
        metadata["timesteps"].append(timestep)

# metadata.json außerhalb speichern
meta_path = os.path.join(os.path.dirname(png_root), "metadata.json")

with open(meta_path, "w") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print(f"Metadata written to {meta_path}")
