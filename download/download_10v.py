#!/usr/bin/env python3
"""
download_10v.py – Lädt '10v' (10v (10m Meridionalwind), Oberfläche, alle Steps) herunter.
"""

import os, time, struct, sys
from ecmwf.opendata import Client

DATE = os.getenv("DATE", "20260325")
TIME = int(os.getenv("RUN", 0))
DATE_ISO = f"{DATE[:4]}-{DATE[4:6]}-{DATE[6:8]}"

STEPS  = list(range(0, 49, 3))
PARAM  = "10v"
FOLDER = os.path.join("data", "gewitter", PARAM)

EXPECTED_MESSAGES = 1   # Oberfläche: 1 Message pro Step
MAX_RETRIES       = 3
RETRY_DELAY       = 10

client = Client(source="aws")


def count_grib2_messages(path: str) -> int:
    count = 0
    try:
        with open(path, "rb") as f:
            while True:
                header = f.read(16)
                if len(header) < 16 or header[:4] != b"GRIB":
                    break
                msg_len = int.from_bytes(header[8:16], "big")
                count += 1
                f.seek(msg_len - 16, 1)
    except Exception:
        return -1
    return count


def is_valid(path: str) -> bool:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False
    n = count_grib2_messages(path)
    if n < EXPECTED_MESSAGES:
        print(f"  ⚠️  {path}: {n} Messages (erwartet ≥{EXPECTED_MESSAGES}) – unvollständig")
        return False
    return True


def download_step(step: int) -> bool:
    os.makedirs(FOLDER, exist_ok=True)
    filename    = f"{PARAM}_step_{step:03d}.grib2"
    target_path = os.path.join(FOLDER, filename)

    if is_valid(target_path):
        print(f"  ✅ Bereits vorhanden und valide: {target_path}")
        return True

    if os.path.exists(target_path):
        os.remove(target_path)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            client.retrieve(
                date=DATE_ISO, time=TIME, type="fc", step=step,
                param=PARAM, levtype="sfc",
                target=target_path,
            )
            if is_valid(target_path):
                print(f"  ✅ {target_path}")
                return True
            print(f"  ⚠️  Versuch {attempt}/{MAX_RETRIES}: Download unvollständig")
        except Exception as e:
            print(f"  ⚠️  Versuch {attempt}/{MAX_RETRIES} Fehler bei Step {step}: {e}")

        if os.path.exists(target_path):
            os.remove(target_path)
        if attempt < MAX_RETRIES:
            print(f"     Warte {RETRY_DELAY}s vor erneutem Versuch...")
            time.sleep(RETRY_DELAY)

    print(f"  ❌ FEHLER: Step {step} nach {MAX_RETRIES} Versuchen fehlgeschlagen!")
    return False


def main():
    print(f"=== Download: {PARAM} (Oberfläche) ===")
    print(f"Datum: {DATE_ISO}  Lauf: {TIME:02d} UTC")
    print()
    failed = [step for step in STEPS if not download_step(step)]
    print()
    if failed:
        print(f"❌ {len(failed)} Steps fehlgeschlagen: {failed}")
        sys.exit(1)
    print(f"✅ Fertig: {PARAM} – alle {len(STEPS)} Steps OK")


if __name__ == "__main__":
    main()
