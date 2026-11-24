"""
Parser for CellASIC ONIX2 experiment files (.OnixExp).

.OnixExp files are ZIP archives containing:
  - Experiment.xml         : plate type, well contents
  - Protocols/Run.protocol.xml : temperature/perfusion protocol steps
  - RunData.txt            : time-series sensor log
  - Protocols/{LoadCells,Prime,Wash}.protocol.xml : standard protocols

Usage:
    from parse_onix import parse_onix_exp
    exp = parse_onix_exp("path/to/file.OnixExp")

    # Or run as a script for a summary:
    python parse_onix.py path/to/file.OnixExp
"""

import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import io
import pandas as pd


NS = "Onix"  # XML namespace


def _tag(name):
    return f"{{{NS}}}{name}"


def _parse_experiment_xml(xml_bytes):
    root = ET.fromstring(xml_bytes)
    plate_info = root.find(_tag("PlateManifoldInfo"))
    result = {
        "plate_type": plate_info.findtext(_tag("PlateType")),
        "manifold_type": plate_info.findtext(_tag("ManifoldType")),
        "plate_definition": plate_info.findtext(_tag("PlateDefinitionFile")),
    }

    # Well contents: {group_index: contents_string}
    wells = {}
    for well in root.findall(f".//{_tag('Well')}"):
        group = int(well.findtext(_tag("WellGroupIndex")))
        unit = int(well.findtext(_tag("UnitIndex")))
        contents = (well.findtext(_tag("Contents")) or "").strip()
        if group not in wells:
            wells[group] = {}
        wells[group][unit] = contents
    result["wells"] = wells
    return result


def _parse_protocol_xml(xml_bytes):
    root = ET.fromstring(xml_bytes)
    steps = []
    for step in root.findall(_tag("Step")):
        step_type = step.get("type") or step.get(
            "{http://www.w3.org/2001/XMLSchema-instance}type", ""
        )
        comment = (step.findtext(_tag("Comment")) or "").strip()

        if "Temperature" in step_type:
            temp_cfg = step.find(_tag("TemperatureConfig"))
            steps.append(
                {
                    "type": "temperature",
                    "comment": comment,
                    "enabled": temp_cfg.findtext(_tag("OffOrOn")) == "True",
                    "temperature_C": float(temp_cfg.findtext(_tag("Temperature"))),
                }
            )
        elif "Perfusion" in step_type:
            duration = int(step.findtext(_tag("Duration")))
            pressure_cfg = step.find(_tag("PressureConfig"))
            pressure_x = float(pressure_cfg.findtext(_tag("PressureX")))
            pressure_y = float(pressure_cfg.findtext(_tag("PressureY")))
            valves = [
                int(v.findtext(_tag("Index")))
                for v in step.findall(f".//{_tag('ValveConfig')}")
            ]
            steps.append(
                {
                    "type": "perfusion",
                    "comment": comment,
                    "duration_s": duration,
                    "duration_min": duration / 60,
                    "pressure_x_psi": pressure_x,
                    "pressure_y_psi": pressure_y,
                    "valves": valves,
                }
            )
    return steps


def _parse_rundata(txt_bytes):
    text = txt_bytes.decode("utf-8", errors="replace")
    lines = text.splitlines()

    # Parse metadata lines (start with ###) from anywhere in the file
    meta = {}
    data_start = 0

    def _parse_meta_line(line):
        parts = line[4:].split("\t")
        key = parts[0].strip()
        values = parts[1:]
        if key == "Acquisition Started" and len(values) == 6:
            y, mo, d, h, mi, s = (int(v) for v in values)
            meta["acquisition_start"] = datetime(y, mo, d, h, mi, s)
        elif key == "Acquisition Ended" and len(values) == 6:
            y, mo, d, h, mi, s = (int(v) for v in values)
            meta["acquisition_end"] = datetime(y, mo, d, h, mi, s)
        elif key == "Version":
            meta["version"] = values[0].strip() if values else ""
        elif key == "Instrument Serial Number":
            meta["instrument_serial"] = values[0].strip() if values else ""

    for i, line in enumerate(lines):
        if line.startswith("### "):
            _parse_meta_line(line)
        elif not line.startswith("###"):
            if data_start == 0:
                data_start = i

    # Find the header row (tab-separated, not starting with ###)
    header_line = None
    first_data = None
    for i in range(data_start, len(lines)):
        line = lines[i]
        if line.startswith("###"):
            continue
        if header_line is None:
            header_line = i
        else:
            first_data = i
            break

    if header_line is None:
        return meta, pd.DataFrame()

    # Read as TSV
    tsv_text = "\n".join(
        l for l in lines[header_line:] if l and not l.startswith("###")
    )
    df = pd.read_csv(
        io.StringIO(tsv_text),
        sep="\t",
        dtype={"TimeStamp": float},
        na_values=[""],
    )

    # Keep useful columns, rename for clarity
    rename = {
        "TimeStamp": "time_s",
        "Step": "protocol_step",
        "X": "pressure_x_psi",
        "Y": "pressure_y_psi",
        "Temperature": "temperature_C",
        "RunState": "run_state",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # Add wall-clock time if acquisition start is known
    if "acquisition_start" in meta:
        df["datetime"] = meta["acquisition_start"] + pd.to_timedelta(
            df["time_s"], unit="s"
        )

    return meta, df


def parse_onix_exp(filepath):
    """
    Parse a CellASIC ONIX2 .OnixExp file.

    Returns a dict with keys:
      plate_type, manifold_type, plate_definition  (str)
      wells          : {group_index: {unit_index: contents_str}}
      run_protocol   : list of step dicts (type, temperature_C / duration_s, valves, ...)
      run_data_meta  : dict (acquisition_start, acquisition_end, instrument_serial, version)
      run_data       : pandas DataFrame (time_s, protocol_step, pressure_x/y_psi,
                       temperature_C, run_state, datetime)
    """
    with zipfile.ZipFile(filepath) as zf:
        names = zf.namelist()

        exp = _parse_experiment_xml(zf.read("Experiment.xml"))

        run_protocol = []
        if "Protocols/Run.protocol.xml" in names:
            run_protocol = _parse_protocol_xml(zf.read("Protocols/Run.protocol.xml"))

        run_data_meta, run_data_df = {}, pd.DataFrame()
        if "RunData.txt" in names:
            run_data_meta, run_data_df = _parse_rundata(zf.read("RunData.txt"))

    return {
        "plate_type": exp["plate_type"],
        "manifold_type": exp["manifold_type"],
        "plate_definition": exp["plate_definition"],
        "wells": exp["wells"],
        "run_protocol": run_protocol,
        "run_data_meta": run_data_meta,
        "run_data": run_data_df,
    }


def summarize(exp):
    """Print a human-readable summary of a parsed experiment."""
    print(f"Plate: {exp['plate_type']}  |  Manifold: {exp['manifold_type']}")

    meta = exp["run_data_meta"]
    if "acquisition_start" in meta:
        print(f"Started:  {meta['acquisition_start']}")
    if "acquisition_end" in meta:
        dur = meta["acquisition_end"] - meta["acquisition_start"]
        print(f"Ended:    {meta['acquisition_end']}  ({dur})")
    if "instrument_serial" in meta:
        print(f"Instrument S/N: {meta['instrument_serial']}")

    print("\nWell contents:")
    for group, units in sorted(exp["wells"].items()):
        contents = {u: c for u, c in units.items() if c}
        if contents:
            label = ", ".join(f"unit {u}: {c}" for u, c in sorted(contents.items()))
            print(f"  Group {group}: {label}")

    print("\nRun protocol:")
    current_temp = None
    for i, step in enumerate(exp["run_protocol"], 1):
        if step["type"] == "temperature":
            current_temp = step["temperature_C"]
            print(f"  Step {i}: Set temperature → {current_temp}°C")
        elif step["type"] == "perfusion":
            dur_h = step["duration_s"] / 3600
            valves = step["valves"]
            print(
                f"  Step {i}: Perfuse {dur_h:.1f}h @ {step['pressure_x_psi']} psi, "
                f"valve(s) {valves}, T={current_temp}°C"
            )

    df = exp["run_data"]
    if not df.empty:
        print(f"\nRun data: {len(df)} rows, time {df['time_s'].min():.0f}–{df['time_s'].max():.0f} s")
        print(f"  Temperature range: {df['temperature_C'].min():.1f}–{df['temperature_C'].max():.1f}°C")
        print(f"  Pressure X range:  {df['pressure_x_psi'].min():.1f}–{df['pressure_x_psi'].max():.1f} psi")


if __name__ == "__main__":
    import sys
    from pathlib import Path

    if len(sys.argv) < 2:
        print("Usage: python parse_onix.py <file.OnixExp>")
        sys.exit(1)

    filepath = sys.argv[1]
    exp = parse_onix_exp(filepath)
    summarize(exp)

    df = exp["run_data"]
    if not df.empty:
        csv_path = Path(filepath).with_suffix(".run_data.csv")
        # Write only the informative columns
        cols = [c for c in ["datetime", "time_s", "protocol_step",
                             "temperature_C", "pressure_x_psi", "pressure_y_psi",
                             "run_state"] if c in df.columns]
        df[cols].to_csv(csv_path, index=False)
        print(f"\nRun data saved to: {csv_path}")
