import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
from scipy.stats import weibull_min

def fit_power_curve(wind_speed,power_output, v_fit = np.linspace(0, 20, 100)):
    def power_curve(v, P_rated, v_ci, v_r, k):
        """Cubic-based power curve for curve fitting."""
        v = np.array(v)
        P = np.zeros_like(v)
        # Between cut-in and rated speed
        mask = (v >= v_ci) & (v < v_r)
        P[mask] = P_rated * ((v[mask] - v_ci) / (v_r - v_ci))**k
        # Above rated speed
        P[v >= v_r] = P_rated
        return P
    popt, pcov = curve_fit(power_curve, wind_speed, power_output, p0=[1, 3, 10, 3])
    P_rated_fit = power_curve(v_fit, *popt)
    return P_rated_fit

def fit_weibull(wind_speed, v_fit = np.linspace(0, 20, 100)):
    shape, loc, scale = weibull_min.fit(wind_speed, floc=0)  # fix loc=0
    pdf_fitted = weibull_min.pdf(v_fit, shape, loc=0, scale=scale)
    return pdf_fitted

def get_loss(P_rated_fit,P_rated_fit_ref,pdf_fitted,v_fit):
    expected_power = P_rated_fit * pdf_fitted
    expected_power_ref = P_rated_fit_ref * pdf_fitted
    # Integrate using trapezoidal rule
    aep = 8760 * np.trapezoid(expected_power, v_fit)  
    aep_ref = 8760 * np.trapezoid(expected_power_ref, v_fit)  
    return 1-aep / aep_ref

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