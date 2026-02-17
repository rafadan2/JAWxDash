import pandas as pd
import io
import re
import numpy as np
import logging


# Local imports
from src.ellipsometry_toolbox.statistics import get_statistics



logger = logging.getLogger(__name__)



def _read_scan_points(lines:list[str]) -> pd.DataFrame|None:
    """
    Parse scan-point section files and return x/y coordinates.

    Expected markers:
    - start_Scan Points
    - end_Scan Points
    """

    if not lines:
        return None

    start_idx = None
    end_idx = None
    for idx, line in enumerate(lines):
        token = line.strip().lower()
        if token == "start_scan points":
            start_idx = idx + 1
            continue
        if token == "end_scan points" and start_idx is not None:
            end_idx = idx
            break

    if start_idx is None or end_idx is None or start_idx >= end_idx:
        return None

    points = []
    for line in lines[start_idx:end_idx]:
        stripped = line.strip()
        if not stripped:
            continue

        values = []
        for token in re.split(r"\s+", stripped):
            try:
                values.append(float(token))
            except ValueError:
                continue

        if len(values) >= 2:
            points.append((values[0], values[1]))

    scan_df = pd.DataFrame(points, columns=["x", "y"], dtype=float)
    scan_df.attrs["source_format"] = "scan_points"
    return scan_df


def read_data(filepath_or_stream:str|bytes) -> pd.DataFrame:
    """
    Read the jaw.TXT file from at filepath or a stream

    Returns a pd.DataFrame with columns:
    - Point #	
    - Z Align	
    - SigInt	
    - Tilt X	
    - Tilt Y	
    - Hardware OK	
    - MSE	
    - Thickness # 1 (nm)	
    - A	
    - B	
    - n of Cauchy @ 632.8 nm	
    - Fit OK
    - x
    - y

    """


    # Determines if filepath or stream
    lines = []
    filepath_or_buffer = ""
    if isinstance(filepath_or_stream, str):
        # is a path
        with open(filepath_or_stream, "r") as f:
            lines = f.readlines()
        filepath_or_buffer = filepath_or_stream

    
    elif isinstance(filepath_or_stream, bytes):
        # is a stream
        buffer = io.StringIO(filepath_or_stream.decode("utf-8"))
        lines = buffer.readlines()

        filepath_or_buffer = io.StringIO(filepath_or_stream.decode("utf-8"))
    

    else:
        logger.info("Expected filepath or stream got type: %s" % type(filepath_or_stream))
        return pd.DataFrame()

    scan_data = _read_scan_points(lines)
    if scan_data is not None:
        logger.info("Detected scan-points file format with %d points.", len(scan_data.index))
        return scan_data

    # Find lines with the data, by matching (decimal,decimal)

    # Pattern explanation
    # \( and \): Match the parentheses that enclose the two numbers.
    # [+-]?: Matches an optional + or - sign before each number.
    # \d+\.\d+: Matches a decimal number (one or more digits before and after the decimal point).
    # ,: Matches the comma separating the two numbers.
    pattern = r"\([+-]?\d+(\.\d+)?,[+-]?\d+(\.\d+)?\)"

    data_line = False
    for i, line in enumerate(lines):
        matches = re.findall(pattern, line)
        if matches:
            data_line = i
            break

    # Reading header
    data = pd.read_csv(filepath_or_buffer, delimiter="\t", header=0, skiprows=range(1, data_line))


    # Extract x and y
    # Pattern check against "(x.x,y.y)"
    pattern = r"[-+]?(?:\d*\.*\d+)"

    x, y = [], []
    for i, xy in enumerate(data.iloc[:, 0].values.tolist()):
        matches = re.findall(pattern, xy)

        if len(matches) == 2:
            x.append(float(matches[0]))
            y.append(float(matches[1]))
        
        else:
            x.append(np.nan)
            y.append(np.nan)

            print(f"Bad pattern! row: {i}, string: {xy}, match: {matches}")

    # Adding x and y to DataFrame
    data['x'] = x
    data['y'] = y
    data.drop(data.columns[0], axis=1, inplace=True)  # drops first column with string (x.xxx, y.yyy) coordinates
    data.attrs["source_format"] = "jaw_table"

    return data



def to_buffer(df:pd.DataFrame) -> io.StringIO:
    """
    Creates a buffered version of the dataframe formatted similar to 
    the JAW.txt file.
    
    NOTE: Column names are not checked
    """

    buffer = io.StringIO()
    
    ### Header generation ###
    # Index that'll be exported
    index_mapping = {
        "mean": "Average",
        "min": "Min",
        "max": "Max",
        "std": "Std. Dev.",
        "% Range": "% Range",
        "% Uniformity": "% Uniformity"
    }

    exp_index = list(index_mapping.values())
    
    stats = get_statistics(df.drop(labels=["x", "y"], axis=1))  # Drops x and y column and generating statistics
    stats = stats.rename(index=index_mapping).loc[exp_index]  # Rename index and pulls desired index
    stats.to_csv(buffer, sep="\t", float_format="%.4f", header=True, index=True)  # writes stats to buffer


    ### Data generation ###
    xy_col = df.apply(lambda row: "(%.3f,%.3f)" % (row.x, row.y), axis=1)
    df.insert(0, "xy", xy_col)
    df.drop(labels=["x", "y"], axis=1, inplace=True)  # drops 'x' and 'y' column
    df.to_csv(buffer, sep="\t", header=False, index=False)

    
    return buffer
