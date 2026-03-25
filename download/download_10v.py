#!/usr/bin/env python3
"""
download_10v.py – Lädt '10v' (Oberfläche, alle Steps) herunter.
Völlig eigenständig, keine externen Abhängigkeiten außer ecmwf-opendata.
"""

import os
from ecmwf.opendata import Client

# --- Laufzeit-Infos aus Umgebungsvariablen ---
DATE = os.getenv("DATE", "20260325")
TIME = int(os.getenv("RUN", 0))        # 0, 6, 12 oder 18

DATE_ISO = f"{DATE[:4]}-{DATE[4:6]}-{DATE[6:8]}"

STEPS  = list(range(0, 49, 3))        # 0–48h alle 3h
PARAM  = "10v"
FOLDER = os.path.join("data", "gewitter", PARAM)

client = Client(source="aws")


def download_step(step: int):
    os.makedirs(FOLDER, exist_ok=True)
    filename    = f"{PARAM}_step_{step:03d}.grib2"
    target_path = os.path.join(FOLDER, filename)

    if os.path.exists(target_path):
        print(f"  ✅ Bereits vorhanden: {target_path}")
        return

    try:
        client.retrieve(
            date=DATE_ISO,
            time=TIME,
            type="fc",
            step=step,
            param=PARAM,
            levtype="sfc",
            target=target_path,
        )
        print(f"  ✅ {target_path}")
    except Exception as e:
        print(f"  ⚠️  Fehler bei Step {step}: {e}")
        if os.path.exists(target_path) and os.path.getsize(target_path) == 0:
            os.remove(target_path)


def main():
    print(f"=== Download: {PARAM} (Oberfläche) ===")
    print(f"Datum: {DATE_ISO}  Lauf: {TIME:02d} UTC")
    print(f"Zielordner: {FOLDER}/")
    print(f"Steps: {STEPS}")
    print()
    for step in STEPS:
        download_step(step)
    print()
    print(f"✅ Fertig: {PARAM}")


if __name__ == "__main__":
    main()
