"""Record tap attempts from the Arduino and append them to a CSV file.

Each row records who tapped and whose rhythm they were trying to tap:
    tapper   = the person at the button
    rhythm   = the owner of the rhythm being tapped
A row with tapper == rhythm is a genuine attempt; anything else is an
impostor (for example, a friend copying your rhythm after watching you).

Usage:
    pip install pyserial
    python collect.py --port /dev/tty.usbmodem1101 --tapper emily --rhythm emily
    python collect.py --port COM3 --tapper alex --rhythm emily   # copycat attempts
"""
import argparse
import csv
import os

import serial  # pyserial


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", required=True, help="serial port of the Arduino")
    p.add_argument("--tapper", required=True)
    p.add_argument("--rhythm", required=True, help="owner of the rhythm being tapped")
    p.add_argument("--out", default="taps.csv")
    p.add_argument("--baud", type=int, default=115200)
    args = p.parse_args()

    new_file = not os.path.exists(args.out)
    with serial.Serial(args.port, args.baud, timeout=1) as ser, open(args.out, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["tapper", "rhythm", "gaps_ms"])
        print(f"Recording {args.tapper} tapping {args.rhythm}'s rhythm. Ctrl-C to stop.")
        count = 0
        try:
            while True:
                line = ser.readline().decode(errors="ignore").strip()
                if not line.startswith("TAPS"):
                    continue
                gaps = line.split(",")[1:]
                if not gaps:
                    continue
                writer.writerow([args.tapper, args.rhythm, ";".join(gaps)])
                f.flush()
                count += 1
                print(f"  attempt {count}: {len(gaps) + 1} taps, gaps = {gaps}")
        except KeyboardInterrupt:
            print(f"\nSaved {count} attempts to {args.out}")


if __name__ == "__main__":
    main()
