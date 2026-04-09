import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import numpy as np
from scipy.stats import weibull_min


def fit_power_curve(wind_speed, power_output, v_fit=np.linspace(0, 20, 100), P_rated = 1):
    def power_curve(v, v_ci, v_r, k):
        """Cubic-based power curve for curve fitting."""
        v = np.array(v)
        P = np.zeros_like(v)
        # Between cut-in and rated speed
        mask = (v >= v_ci) & (v < v_r)
        P[mask] = P_rated * ((v[mask] - v_ci) / (v_r - v_ci)) ** k
        # Above rated speed
        P[v >= v_r] = P_rated
        return P

    popt, pcov = curve_fit(power_curve, wind_speed, power_output, p0=[3, 10, 3])
    P_rated_fit = power_curve(v_fit, *popt)
    return P_rated_fit


def fit_weibull(wind_speed, v_fit=np.linspace(0, 20, 100)):
    shape, loc, scale = weibull_min.fit(wind_speed, floc=0)  # fix loc=0
    pdf_fitted = weibull_min.pdf(v_fit, shape, loc=0, scale=scale)
    return pdf_fitted


def get_loss(P_rated_fit, P_rated_fit_ref, pdf_fitted, v_fit):
    expected_power = P_rated_fit * pdf_fitted
    expected_power_ref = P_rated_fit_ref * pdf_fitted
    # Integrate using trapezoidal rule
    aep = 8760 * np.trapezoid(expected_power, v_fit)
    aep_ref = 8760 * np.trapezoid(expected_power_ref, v_fit)
    return 1 - aep / aep_ref


def plot_pc(
    wind_speed,
    power_output,
    wind_speed_ref,
    power_output_ref,
    title: str,
    v_fit=np.linspace(0, 20, 100),
    reveres_zorder=False,
    plot_pc_fit=False,
    P_rated = 1,
    P_rated_ref = 1,
):
    plt.figure(figsize=(10, 6))
    pdf_fitted = fit_weibull(np.hstack([wind_speed, wind_speed_ref]), v_fit)

    if reveres_zorder:
        zorder_observed, zorder_ref = 2, 1
    else:
        zorder_observed, zorder_ref = 1, 2

    plt.scatter(
        wind_speed,
        power_output,
        s=5,
        alpha=0.5,
        label="Observed Power Output",
        zorder=zorder_observed,
    )
    plt.scatter(
        wind_speed_ref,
        power_output_ref,
        s=5,
        alpha=0.5,
        label="Reference Power Output",
        zorder=zorder_ref,
    )
    plt.plot(v_fit, pdf_fitted, "r-", lw=2, label="Fitted Weibull PDF")

    plt.xlabel("Wind Speed (m/s)")
    plt.ylabel("Value (Normalized Power Curve \n& Weibull Probability Density)")
    plt.title(title)
    
    if plot_pc_fit:
        P_rated_fit = fit_power_curve(wind_speed, power_output, v_fit, P_rated = P_rated)
        P_rated_fit_ref = fit_power_curve(wind_speed_ref, power_output_ref, v_fit, P_rated = P_rated_ref)
        plt.plot(
            v_fit,
            P_rated_fit,
            label="Fitted Curve",
            color="lime",
            linewidth=2,
            zorder=2,
        )
        plt.plot(
            v_fit,
            P_rated_fit_ref,
            label="Fitted Curve Reference",
            color="black",
            linewidth=2,
            zorder=3,
        )
        print(f"Loss = {get_loss(P_rated_fit, P_rated_fit_ref, pdf_fitted, v_fit):.3f}")
    plt.legend()
    plt.grid()
    file_name = title.replace(" ", "_")
    plt.savefig(f"plots/{file_name}.png")

    # print(f"Loss = {get_loss(P_rated_fit,P_rated_fit_ref,pdf_fitted,v_fit):.3f}")
