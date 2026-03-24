#!/usr/bin/env python3
"""
download_gewitter.py  –  v5

Korrekturen gegenüber v4:
  - "z" auf Druckniveaus → "gh" (Geopotential Height [m], direkt in Metern)
    Ordner: data/gewitter/gh/, Dateien: gh_pl_step_*.grib2
    Kein Division durch g nötig in process.py!
"""

import os
from ecmwf.opendata import Client

# --- Laufzeit-Infos aus Umgebungsvariablen ---
date = os.getenv("DATE", "20260324")   # z. B. "20260324"
time = int(os.getenv("RUN", 0))        # 0, 6, 12 oder 18

if not date:
    raise ValueError("Bitte DATE als Umgebungsvariable setzen, z. B. DATE=20260324")

date_iso = f"{date[:4]}-{date[4:6]}-{date[6:8]}"

client = Client(source="aws")

# Steps: 0–48h alle 3h
steps = list(range(0, 49, 3))

# Druckniveaus laut Open-Data-Katalog
PRESSURE_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

# -------------------------------------------------------
# Oberflächenparameter – alle Steps
# -------------------------------------------------------
PARAMS_SFC = [
    "2t",      # 2m Temperatur
    "2d",      # 2m Taupunkt
    "sp",      # Bodendruck
    "tp",      # Gesamtniederschlag
    "lsm",     # Land-See-Maske
    "mucape",  # Most Unstable CAPE
    "10u",     # 10m Wind U
    "10v",     # 10m Wind V
]

# Orographie: z nur bei Step 0 (zeitkonstant, Geopotential Oberfläche)
PARAMS_SFC_STEP0_ONLY = [
    "z",
]

# -------------------------------------------------------
# Druckniveauparameter – alle Steps
# -------------------------------------------------------
PARAMS_PL = [
    "t",   # Temperatur
    "q",   # Spez. Feuchte
    "r",   # Rel. Feuchte
    "u",   # Wind U
    "v",   # Wind V
    "gh",  # Geopotential Height [m]  → gespeichert als gh_pl_step_*.grib2
]

BASE_DIR = os.path.join("data", "gewitter")


# -------------------------------------------------------
# Download-Funktion
# -------------------------------------------------------

def download_param(param, step, level_type="sfc", levelist=None):
    """Lädt einen einzelnen Parameter für einen Step herunter."""
    folder = os.path.join(BASE_DIR, param)
    os.makedirs(folder, exist_ok=True)

    if level_type == "sfc":
        filename = f"{param}_step_{step:03d}.grib2"
    else:
        filename = f"{param}_pl_step_{step:03d}.grib2"

    target_path = os.path.join(folder, filename)

    if os.path.exists(target_path):
        print(f"  ✅ Bereits vorhanden: {target_path}")
        return

    try:
        kwargs = dict(
        date=date_iso,
        time=time,
        type="fc",
        step=step,
        param=param,
        target=target_path,
    )

        # 🔧 WICHTIG: Level-Typ explizit setzen
        if level_type == "sfc":
            kwargs["levtype"] = "sfc"
        elif level_type == "pl":
            kwargs["levtype"] = "pl"
            if levelist:
                kwargs["levelist"] = levelist

        client.retrieve(**kwargs)
        print(f"  ✅ {target_path}")

    except Exception as e:
        print(f"  ⚠️ Fehler bei {param} Step {step}: {e}")
        if os.path.exists(target_path) and os.path.getsize(target_path) == 0:
            os.remove(target_path)


# -------------------------------------------------------
# Hauptprogramm
# -------------------------------------------------------

def main():
    print("=== Gewitterdaten-Download (v4) ===")
    print(f"Datum: {date_iso}  Lauf: {time:02d} UTC")
    print(f"Zielordner: {BASE_DIR}/")
    print(f"Steps: {steps}")
    print()
    print("Hinweis: deg0l, bli, cin, cp werden NICHT heruntergeladen")
    print("         → werden in process.py aus Druckniveaudaten berechnet")
    print()

    # Orographie nur bei Step 0
    for param in PARAMS_SFC_STEP0_ONLY:
        print(f"--- Lade {param} (Orographie, nur Step 0) ---")
        download_param(param, 0, level_type="sfc")

    # Oberflächenparameter über alle Steps
    for param in PARAMS_SFC:
        print(f"--- Lade {param} (Oberfläche) ---")
        for step in steps:
            download_param(param, step, level_type="sfc")

    # Druckniveauparameter über alle Steps
    for param in PARAMS_PL:
        print(f"--- Lade {param} (Druckniveaus: {PRESSURE_LEVELS}) ---")
        for step in steps:
            download_param(param, step, level_type="pl", levelist=PRESSURE_LEVELS)

    print()
    print("✅ Download abgeschlossen!")
    print()
    print("Fehlende Parameter (werden in process.py berechnet/ersetzt):")
    print("  deg0l  → aus t+gh auf Druckniveaus interpoliert")
    print("  bli    → MU_LI aus theta_ep auf Druckniveaus berechnet")
    print("  mcpr   → aus Δtp/Δt approximiert (kein echtes cp verfügbar)")
    print("  CIN    → nicht berechenbar, wird als NaN gesetzt")


if __name__ == "__main__":
    main()
