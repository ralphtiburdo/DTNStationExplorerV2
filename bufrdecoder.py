#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import eccodes as ec
from eccodes import CodesInternalError

# Try to import the more specific edition error
try:
    from gribapi.errors import UnsupportedEditionError
except ImportError:
    UnsupportedEditionError = None

def decode_bufr_to_dicts(file_path, log_file=None):
    """
    Decode every message in a BUFR file into a list of dicts.
    Each dict is one message; all keys (and their units) are unpacked.
    Unsupported‐edition messages are skipped but reported.
    """
    messages = []
    msg_id = 0

    with open(file_path, "rb") as f:
        while True:
            try:
                bufr = ec.codes_bufr_new_from_file(f)
            except UnsupportedEditionError:
                warn = f"⚠️ Skipping unsupported BUFR edition at message {msg_id} in {os.path.basename(file_path)}"
                print(warn)
                if log_file:
                    with open(log_file, "a") as lf:
                        lf.write(warn + "\n")
                msg_id += 1
                continue
            except Exception:
                break

            if bufr is None:
                break

            try:
                # Get edition number (helps debugging)
                edition = ec.codes_get(bufr, "edition")

                ec.codes_set(bufr, "unpack", 1)
                iterid = ec.codes_bufr_keys_iterator_new(bufr)

                msg = {
                    "file": os.path.basename(file_path),
                    "message_id": msg_id,
                    "edition": edition
                }

                while ec.codes_bufr_keys_iterator_next(iterid):
                    key = ec.codes_bufr_keys_iterator_get_name(iterid)
                    try:
                        val = ec.codes_get(bufr, key)
                    except CodesInternalError:
                        continue

                    # Convert arrays to comma-separated string
                    if isinstance(val, np.ndarray):
                        val = ",".join(map(str, val.tolist()))

                    # Replace missing values with NaN
                    if isinstance(val, float) and val in (-1.0e100, 2147483647):
                        val = np.nan

                    msg[key] = val

                    # Add unit if available
                    try:
                        unit = ec.codes_get(bufr, f"{key}->units")
                        msg[f"{key}_unit"] = unit
                    except CodesInternalError:
                        pass

                ec.codes_bufr_keys_iterator_delete(iterid)

            finally:
                ec.codes_release(bufr)

            messages.append(msg)
            msg_id += 1

    return messages


if __name__ == "__main__":
    # Path with BUFR .bin files
    bufr_dir = "/Users/ralphtiburdo/Downloads/SYNOP BUFR"
    log_file = os.path.join(bufr_dir, "unsupported_messages.log")

    # Collect all .bin files
    files_to_decode = [
        os.path.join(bufr_dir, f)
        for f in os.listdir(bufr_dir)
        if f.lower().endswith(".bin")
    ]

    if not files_to_decode:
        print(f"❌ No .bin files found in {bufr_dir}")
        exit(1)

    for bufr_file in files_to_decode:
        print(f"\n🔎 Decoding {bufr_file} …")
        msgs = decode_bufr_to_dicts(bufr_file, log_file=log_file)
        if not msgs:
            print("⚠️ No messages decoded; skipping output.")
            continue

        df = pd.DataFrame(msgs)

        # Write CSV (same folder as input)
        csv_out = os.path.splitext(os.path.basename(bufr_file))[0] + ".csv"
        csv_out = os.path.join(bufr_dir, csv_out)
        df.to_csv(csv_out, index=False)
        print(f"✅ Wrote CSV → {csv_out} (shape: {df.shape})")