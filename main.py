from simulation.simulator import simulate_engagement
from simulation.scenarios import SCENARIO_REGISTRY
from results.metrics import compute_metrics, compare_metrics
from results.plots import plot_results
from results.output_manager import save_all_outputs


# Available simulation scenarios
SCENARIOS = {
    0: "nominal",
    1: "maneuvering_target",
    2: "acceleration_saturation",
    3: "discrete_guidance",
    4: "variable_acceleration",
    5: "complete",
}

SELECTED_SCENARIO = 5

if SELECTED_SCENARIO not in SCENARIOS:
    raise ValueError(f"Invalid scenario number: {SELECTED_SCENARIO}")

scenario_name = SCENARIOS[SELECTED_SCENARIO]


# Available guidance law combinations
GUIDANCE_LAW_COMBINATIONS = {
    0: ["PP"],
    1: ["TPN"],
    2: ["APN"],
    3: ["PPN"],
    4: ["TPN", "APN", "PPN"],
    5: ["PP", "TPN", "APN", "PPN"],
}

SELECTED_GUIDANCE_LAWS = 5

if SELECTED_GUIDANCE_LAWS not in GUIDANCE_LAW_COMBINATIONS:
    raise ValueError(f"Invalid guidance law combination: {SELECTED_GUIDANCE_LAWS}")

guidance_laws = GUIDANCE_LAW_COMBINATIONS[SELECTED_GUIDANCE_LAWS]


# Output and plotting settings
SAVE_OUTPUTS = True
GENERATE_PLOTS = True
PLOT_LEGEND = True
OUTPUT_DATA = "results/data"
OUTPUT_FIGURES = "results/figures"
FILE_FORMAT = "png"
PLOT_LANGUAGE = "pt"   # "en" or "pt"


# Run one scenario with one guidance law
def run_case(scenario_name, guidance_law):
    if scenario_name not in SCENARIO_REGISTRY:
        raise ValueError(f"Invalid scenario name: {scenario_name}")

    scenario = SCENARIO_REGISTRY[scenario_name](guidance_law=guidance_law)

    results = simulate_engagement(scenario)
    metrics = compute_metrics(results)

    return results, metrics


if __name__ == "__main__":

    results_list = []
    metrics_list = []

    # Run all selected guidance laws for the chosen scenario
    for guidance_law in guidance_laws:
        results, metrics = run_case(scenario_name, guidance_law)

        results_list.append(results)
        metrics_list.append(metrics)

        if SAVE_OUTPUTS:
            results["scenario_key"] = scenario_name

            data_output_directory, summary = save_all_outputs(
                results,
                root_directory=OUTPUT_DATA,
            )

            if GENERATE_PLOTS:
                law_figures_directory = (
                    f"{OUTPUT_FIGURES}/{scenario_name}/laws/{guidance_law.lower()}"
                )

                # Generate plots for the individual guidance law
                plot_results(
                    results,
                    output_directory=law_figures_directory,
                    prefix=guidance_law.lower(),
                    file_format=FILE_FORMAT,
                    show_plots=False,
                    include_closing_velocity=False,
                    language=PLOT_LANGUAGE,
                    show_legend=PLOT_LEGEND,
                )

    # Generate comparison plots when more than one guidance law is simulated
    if SAVE_OUTPUTS and GENERATE_PLOTS and len(results_list) > 1:
        comparison_output_directory = f"{OUTPUT_FIGURES}/{scenario_name}/comparison"

        plot_results(
            results_list,
            output_directory=comparison_output_directory,
            prefix=scenario_name,
            file_format=FILE_FORMAT,
            show_plots=False,
            include_closing_velocity=False,
            language=PLOT_LANGUAGE,
            show_legend=PLOT_LEGEND,
        )

    # Print selected setup and numerical comparison
    print("\n- SELECTED SIMULATION SETUP:")
    print(f"Scenario: {scenario_name}")
    print(f"Guidance laws: {', '.join(guidance_laws)}")
    print(f"Saved outputs: {SAVE_OUTPUTS}")
    print(f"Generated plots: {GENERATE_PLOTS}")

    compare_metrics(metrics_list)