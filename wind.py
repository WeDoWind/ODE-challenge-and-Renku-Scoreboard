import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def powercurve_gain_ref(
    df,
    x_axis="WdSpd - RefHt WEC001 [m/s]",
    y_axis="Power WEC001 Fraction of nomimal power [-]",
    expected_col="Expected Power WEC002 [-]",
    bin_width=0.5,
    ws_min=0,
    ws_max=25,
    min_points_per_bin=5,
    plot=True,
    title = 'Power Curve',
    x_label = "Wind Speed",
    y_label = "Normalized Powercurve",
):
    """
    Compute binned actual vs expected power curve and deviation.
    """

    df = df.copy()

    # Create bins
    bins = np.arange(ws_min, ws_max + bin_width, bin_width)
    labels = bins[:-1] + bin_width / 2

    # Assign bins
    df["ws_bin"] = pd.cut(df[x_axis], bins=bins, labels=labels)

    # Group
    grouped = df.groupby("ws_bin").agg({
        y_axis: ["mean", "count"],
        expected_col: "mean"
    })

    grouped.columns = ["power_actual", "count", "power_expected"]

    result = grouped.reset_index()
    result.rename(columns={"ws_bin": "windspeed_bin"}, inplace=True)

    # Convert bin labels to numeric
    result["windspeed_bin"] = result["windspeed_bin"].astype(float)

    # Filter low data bins
    mask = result["count"] < min_points_per_bin
    result.loc[mask, ["power_actual", "power_expected"]] = np.nan

    # Deviations
    result["abs_diff"] = result["power_actual"] - result["power_expected"]
    result["pct_diff"] = 100 * result["abs_diff"] / result["power_expected"]

    # Plot
    if plot:
        plt.plot(result["windspeed_bin"], result["power_actual"], label="Actual" , c= "r")
        plt.plot(result["windspeed_bin"], result["power_expected"], label="Expected", c= "k")
        plt.xlabel("Wind Speed (m/s)")
        plt.ylabel("Power")
        plt.legend()
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.grid()
    return result


def powercurve_gain(
    df_before,
    df_after,
    ws_col="windspeed",
    power_col="power",
    bin_width=0.5,
    title = 'Power Curve',
    x_label = "Wind Speed",
    y_label = "Normalized Powercurve",
    ws_min=0,
    ws_max=25,
    min_points_per_bin=30,
    plot = True
):
    """
    Create binned power curves and calculate gain/loss between two datasets.

    Parameters:
    - df_before, df_after: pandas DataFrames
    - ws_col: wind speed column name
    - power_col: power column name
    - bin_width: wind speed bin size (m/s)
    - ws_min, ws_max: range of wind speeds
    - min_points_per_bin: minimum samples required per bin

    Returns:
    - DataFrame with:
        windspeed_bin_center
        power_before
        power_after
        abs_gain
        pct_gain
        count_before
        count_after
    """

    # Create bins
    bins = np.arange(ws_min, ws_max + bin_width, bin_width)
    labels = bins[:-1] + bin_width / 2

    def compute_curve(df):
        df = df.copy()
        df["ws_bin"] = pd.cut(df[ws_col], bins=bins, labels=labels)

        grouped = df.groupby("ws_bin")[power_col].agg(["mean", "count"])
        grouped.columns = ["power", "count"]

        return grouped

    curve_before = compute_curve(df_before)
    curve_after = compute_curve(df_after)

    # Merge curves
    result = pd.merge(
        curve_before,
        curve_after,
        left_index=True,
        right_index=True,
        how="outer",
        suffixes=("_before", "_after")
    )

    result = result.reset_index()
    result.rename(columns={"ws_bin": "windspeed_bin"}, inplace=True)

    # Filter bins with low data
    result.loc[result["count_before"] < min_points_per_bin, "power_before"] = np.nan
    result.loc[result["count_after"] < min_points_per_bin, "power_after"] = np.nan

    # Gains
    result["abs_gain"] = result["power_after"] - result["power_before"]
    result["pct_gain"] = 100 * result["abs_gain"] / result["power_before"]
    if plot:
        plt.plot(result["windspeed_bin"], result["power_before"], label = "before")
        plt.plot(result["windspeed_bin"], result["power_after"], label = "after")
        plt.legend()
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.grid()
    return result

def powercurve_scatter(
    df_before,
    df_after,
    ws_col="windspeed",
    power_col="power",
    title = 'Power Curve',
    x_label = "Wind Speed",
    y_label = "Normalized Powercurve",
):


    # Create bins
    plt.scatter(df_before[ws_col], df_before[power_col], s=5, alpha=0.5, label= 'before change')
    plt.scatter(df_after[ws_col], df_after[power_col], s=5, alpha=0.5, label= 'after change')
    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid()
    

def get_powercurve(df, 
                   x_axis, 
                   y_axis, 
                   title = 'Power Curve',
                   x_label = "Wind Speed (m/s)",
                   y_label = "Normalized Powercurve",
                   bin_width=0.5,
                   min_count=5,
                   expected_col = None,
                   smooth=True,
                   **kwargs,
                   ):


    # Drop NaNs
    data = df[[x_axis, y_axis]].dropna()

    # Create bins
    bins = np.arange(data[x_axis].min(), data[x_axis].max() + bin_width, bin_width)
    data['bin'] = pd.cut(data[x_axis], bins)

    # Aggregate
    grouped = data.groupby('bin').agg(
        wind_speed_mean=(x_axis, 'mean'),
        power_mean=(y_axis, 'mean'),
        count=(y_axis, 'count')
    ).reset_index()

    # Filter sparse bins
    grouped = grouped[grouped['count'] >= min_count]

    # Optional smoothing
    if smooth:
        grouped['power_mean'] = grouped['power_mean'].rolling(
            window=3, center=True, min_periods=1
        ).mean()

    # Plot
    # plt.figure(figsize=(8, 5))
    plt.scatter(data[x_axis], data[y_axis], s=5, alpha=0.5, label= 'Observed Power Output')

    if expected_col is not None:
        plt.scatter(df[x_axis], df[expected_col], s=5, alpha=0.5,label = "Reference Power Curve")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    # plt.savefig(f"{title}.png")