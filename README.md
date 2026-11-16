# Tap-rhythm lock (CS109 final project starter)

## Hardware
- Arduino (Uno, Nano, or similar) and one push button.
- Wire the button between **pin 2** and **GND**. No resistor is needed, because the sketch uses the internal pull-up.

## Steps
1. Upload `tap_recorder/tap_recorder.ino`. In the Serial Monitor (115200 baud), tap a rhythm and pause. You should see a line like `TAPS,182.4,179.0,361.2,...`. Close the Serial Monitor before running `collect.py`, since only one program can use the port at a time.
2. Record data (`pip install pyserial numpy scipy matplotlib`):
   ```
   python collect.py --port <PORT> --tapper emily --rhythm emily   # 30-50 genuine attempts
   python collect.py --port <PORT> --tapper alex  --rhythm emily   # 10-15 per copycat
   ```
   Pick a rhythm of about 8 taps with uneven gaps. Let copycats watch and listen to you before they try.
3. Analyze:
   ```
   python analyze.py taps.csv
   python analyze.py --simulate    # try it now with synthetic data
   ```
   This prints AUC and equal error rate, and saves `tap_lock_results.png`.

## Options worth experimenting with
- `--raw`: use raw gaps instead of log-gaps, and compare both the normality tests and the ROC curves.
- `--shrink 0` to `1`: how much the covariance is regularized. Try it with only 5–10 genuine attempts.
