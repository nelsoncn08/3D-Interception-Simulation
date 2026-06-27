from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def plot_pursuer_longitudinal_acceleration_profile(
    scenario,
    t_final=60.0,
    dt=0.01,
    output_file="results/figures/input_profiles/pursuer_longitudinal_acceleration.png",
    show_plot=False,
):
    acceleration_function = scenario.get("pursuer_acceleration_function", None)

    if not callable(acceleration_function):
        raise ValueError("Scenario does not contain a valid pursuer_acceleration_function.")

    time = np.arange(0.0, t_final + dt, dt)
    acceleration = np.array([acceleration_function(t) for t in time], dtype=float)

    plt.figure(figsize=(7, 4))
    plt.plot(time, acceleration)
    plt.axhline(0.0, linestyle="--")

    plt.xlabel("Time [s]")
    plt.ylabel("Longitudinal acceleration [m/s²]")
    plt.title("Prescribed pursuer longitudinal acceleration")
    plt.grid(True)

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")

    if show_plot:
        plt.show()

    plt.close()

    return output_file