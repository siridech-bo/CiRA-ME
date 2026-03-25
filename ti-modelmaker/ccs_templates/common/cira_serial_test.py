#!/usr/bin/env python3
"""
CiRA ME - Serial Test Tool for TI MCU

This script communicates with a TI C2000 MCU running the CiRA ME
inference firmware. It sends feature vectors via UART and reads
predictions back.

Usage:
    python cira_serial_test.py --port COM5
    python cira_serial_test.py --port COM5 --csv test_data.csv
    python cira_serial_test.py --port COM5 --interactive

Requirements:
    pip install pyserial numpy

Protocol:
    PC -> MCU:  [0xAA] [num_features:uint8] [f1:float32] ... [fN:float32]
    MCU -> PC:  [0xBB] [prediction:float32] [inference_time_us:uint32]
"""

import argparse
import struct
import sys
import time

try:
    import serial
except ImportError:
    print("ERROR: pyserial not installed. Run: pip install pyserial")
    sys.exit(1)

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# Protocol constants
PROTO_START_INPUT  = 0xAA
PROTO_START_OUTPUT = 0xBB
PROTO_CMD_PING     = 0x01
PROTO_CMD_INFO     = 0x02


def connect(port, baudrate=115200, timeout=2):
    """Connect to MCU via serial port."""
    try:
        ser = serial.Serial(port, baudrate, timeout=timeout)
        time.sleep(0.5)  # Wait for MCU reset
        # Flush any startup messages
        ser.read(ser.in_waiting)
        return ser
    except serial.SerialException as e:
        print(f"ERROR: Cannot connect to {port}: {e}")
        sys.exit(1)


def ping(ser):
    """Send ping and check MCU responds."""
    ser.write(bytes([PROTO_CMD_PING]))
    time.sleep(0.1)
    response = ser.read(ser.in_waiting).decode('ascii', errors='ignore')
    if 'CiRA-ME-OK' in response:
        print(f"  MCU responded: {response.strip()}")
        return True
    else:
        print(f"  No valid response (got: {repr(response)})")
        return False


def send_features(ser, features):
    """Send feature vector and receive prediction."""
    n = len(features)

    # Build packet: [0xAA] [num_features] [f1] ... [fN]
    packet = bytes([PROTO_START_INPUT, n & 0xFF])
    for f in features:
        packet += struct.pack('<f', float(f))

    ser.write(packet)

    # Read response: [0xBB] [prediction:float32] [time_us:uint32]
    response = ser.read(9)  # 1 + 4 + 4 bytes
    if len(response) < 9:
        return None, None

    if response[0] != PROTO_START_OUTPUT:
        return None, None

    prediction = struct.unpack('<f', response[1:5])[0]
    inference_us = struct.unpack('<I', response[5:9])[0]

    return prediction, inference_us


def test_single(ser, features):
    """Test with a single feature vector."""
    print(f"\nSending {len(features)} features: {features}")
    pred, time_us = send_features(ser, features)
    if pred is not None:
        print(f"  Prediction:     {pred:.6f}")
        print(f"  Inference time: {time_us} us ({time_us/1000:.2f} ms)")
    else:
        print("  ERROR: No valid response from MCU")
    return pred, time_us


def test_csv(ser, csv_path, target_col=None):
    """Test with CSV file, compare predictions if target column exists."""
    if not HAS_NUMPY:
        print("ERROR: numpy required for CSV testing. Run: pip install numpy")
        return

    import csv

    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        rows = [row for row in reader]

    print(f"\nCSV: {csv_path}")
    print(f"Columns: {headers}")
    print(f"Rows: {len(rows)}")

    # Find target column
    target_idx = None
    if target_col and target_col in headers:
        target_idx = headers.index(target_col)
        print(f"Target column: {target_col} (index {target_idx})")

    # Determine feature columns (exclude non-numeric, timestamp, target)
    feature_indices = []
    for i, h in enumerate(headers):
        if i == target_idx:
            continue
        try:
            float(rows[0][i])
            feature_indices.append(i)
        except (ValueError, IndexError):
            pass

    print(f"Feature columns ({len(feature_indices)}): "
          f"{[headers[i] for i in feature_indices]}")

    # Run predictions
    predictions = []
    actuals = []
    times = []

    n_test = min(len(rows), 100)  # Limit to 100 rows
    print(f"\nRunning {n_test} predictions...")

    for row_idx in range(n_test):
        row = rows[row_idx]
        features = [float(row[i]) for i in feature_indices]
        pred, time_us = send_features(ser, features)

        if pred is not None:
            predictions.append(pred)
            times.append(time_us)
            if target_idx is not None:
                actuals.append(float(row[target_idx]))

            if (row_idx + 1) % 10 == 0:
                print(f"  {row_idx + 1}/{n_test} done "
                      f"(avg inference: {np.mean(times):.0f} us)")

    # Report results
    print(f"\n{'='*50}")
    print(f"Results: {len(predictions)} predictions")
    print(f"  Avg inference time: {np.mean(times):.0f} us "
          f"({np.mean(times)/1000:.2f} ms)")
    print(f"  Min/Max time: {np.min(times)}/{np.max(times)} us")
    print(f"  Prediction range: [{np.min(predictions):.4f}, "
          f"{np.max(predictions):.4f}]")

    if actuals:
        actuals = np.array(actuals)
        preds = np.array(predictions)
        residuals = actuals - preds

        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        rmse = np.sqrt(np.mean(residuals ** 2))
        mae = np.mean(np.abs(residuals))

        print(f"\n  Metrics (vs {target_col}):")
        print(f"    R2:   {r2:.4f}")
        print(f"    RMSE: {rmse:.4f}")
        print(f"    MAE:  {mae:.4f}")

    # Save results
    out_path = csv_path.replace('.csv', '_mcu_results.csv')
    with open(out_path, 'w') as f:
        f.write('prediction,inference_us')
        if actuals:
            f.write(',actual,residual')
        f.write('\n')
        for i in range(len(predictions)):
            f.write(f'{predictions[i]:.6f},{times[i]}')
            if actuals:
                f.write(f',{actuals[i]:.6f},{actuals[i]-predictions[i]:.6f}')
            f.write('\n')
    print(f"\nResults saved to: {out_path}")


def interactive_mode(ser, num_features):
    """Interactive mode: enter features manually."""
    print(f"\nInteractive mode ({num_features} features)")
    print("Enter features separated by spaces, or 'q' to quit\n")

    while True:
        try:
            line = input(f"Features ({num_features} values): ").strip()
            if line.lower() in ('q', 'quit', 'exit'):
                break

            values = [float(x) for x in line.split()]
            if len(values) != num_features:
                print(f"  Expected {num_features} values, got {len(values)}")
                continue

            test_single(ser, values)
        except ValueError:
            print("  Invalid input. Enter space-separated numbers.")
        except KeyboardInterrupt:
            break

    print("\nBye!")


def main():
    parser = argparse.ArgumentParser(
        description='CiRA ME - TI MCU Serial Test Tool')
    parser.add_argument('--port', required=True,
                        help='Serial port (e.g., COM5, /dev/ttyACM0)')
    parser.add_argument('--baud', type=int, default=115200,
                        help='Baud rate (default: 115200)')
    parser.add_argument('--features', type=int, default=6,
                        help='Number of features (default: 6)')
    parser.add_argument('--csv', type=str,
                        help='CSV file for batch testing')
    parser.add_argument('--target', type=str,
                        help='Target column name in CSV for evaluation')
    parser.add_argument('--interactive', action='store_true',
                        help='Interactive mode: enter features manually')
    parser.add_argument('--test', nargs='+', type=float,
                        help='Single test: --test 1.0 2.0 3.0')

    args = parser.parse_args()

    print("=" * 50)
    print("CiRA ME - TI MCU Serial Test Tool")
    print("=" * 50)
    print(f"Port: {args.port}")
    print(f"Baud: {args.baud}")

    ser = connect(args.port, args.baud)
    print(f"Connected to {args.port}")

    # Ping device
    print("\nPinging MCU...")
    if not ping(ser):
        print("WARNING: MCU did not respond to ping. "
              "Check connection and firmware.")

    if args.csv:
        test_csv(ser, args.csv, args.target)
    elif args.test:
        test_single(ser, args.test)
    elif args.interactive:
        interactive_mode(ser, args.features)
    else:
        # Default: ping and show info
        print("\nUse --csv, --test, or --interactive for testing")
        print("Examples:")
        print(f"  python {sys.argv[0]} --port {args.port} --test 1.0 2.0 3.0 4.0 5.0 6.0")
        print(f"  python {sys.argv[0]} --port {args.port} --csv test_data.csv --target Pressure_PSI")
        print(f"  python {sys.argv[0]} --port {args.port} --interactive --features 6")

    ser.close()


if __name__ == '__main__':
    main()
