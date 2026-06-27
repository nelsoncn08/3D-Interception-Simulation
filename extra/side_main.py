from simulation.scenarios import variable_acceleration

from extra.complementary_functions import (
    plot_pursuer_longitudinal_acceleration_profile,
)


RUN_LONGITUDINAL_ACCELERATION_PROFILE = True


if __name__ == "__main__":
    if RUN_LONGITUDINAL_ACCELERATION_PROFILE:
        scenario = variable_acceleration(guidance_law="TPN")

        output_file = plot_pursuer_longitudinal_acceleration_profile(
            scenario=scenario,
            t_final=60.0,
            dt=0.01,
            output_file="results/figures/input_profiles/pursuer_longitudinal_acceleration.png",
            show_plot=False,
        )

        print(f"Longitudinal acceleration profile saved to: {output_file}")