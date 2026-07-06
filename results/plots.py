from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from results.language import DEFAULT_LANGUAGE, USE_FIGURE_TITLES, get_text

from results.plot_config import (
    LAW_COLORS,
    TARGET_COLOR,
    LIMIT_COLOR,
    LIMIT_LINE,
    LINE_WIDTH,
    LIMIT_LINE_WIDTH,
    MARKER_BIG,
    MARKER_MEDIUM,
    MARKER_SMALL,
    AXIS_LABEL_SIZE,
    TICK_LABEL_SIZE,
    LEGEND_SIZE,
    FIGURE_SIZE_2D,
    FIGURE_SIZE_3D,
    GRID,
    DPI,
    REMOVE_PLOT_SPIKES,
    PLOT_SPIKE_FACTOR,
    PLOT_SPIKE_REFERENCE,
)


# Return the color associated with each guidance law
def _guidance_color(results):
    guidance_law = results.get("guidance_law", "Unknown").upper()
    return LAW_COLORS.get(guidance_law, "tab:purple")


# Bug fix: ensure that the plotting functions always receive a list of results
def _as_list(results):
    if isinstance(results, list):
        return results

    return [results]


# Return the guidance law name used in legends
def _guidance_label(results):
    return results.get("guidance_law", "Unknown")


# Return the magnitude history of scalar or vector
def _norm_history(vector_history):
    vector_history = np.asarray(vector_history, dtype=float)

    if vector_history.size == 0:
        return np.array([])

    if vector_history.ndim == 1:
        return np.abs(vector_history)

    return np.linalg.norm(vector_history, axis=1)


# Check if a given history exists and has nonzero values
def _has_nonzero_history(results, key, tolerance=1e-12):
    for result in _as_list(results):
        if key not in result:
            continue

        values = np.asarray(result[key], dtype=float)

        if values.size > 0 and np.any(np.abs(values) > tolerance):
            return True

    return False


# Remove visually excessive spikes from the plots
def _remove_plot_spikes(values, reference_limit=None, factor=10.0):
    values = np.asarray(values, dtype=float).copy()

    if values.size == 0:
        return values

    if reference_limit is None or np.isinf(reference_limit):
        return values

    plot_limit = factor * reference_limit

    values[np.abs(values) > plot_limit] = np.nan

    return values


# Create the output file path when an output directory is provided
def _make_output_file(output_directory, filename):
    if output_directory is None:
        return None

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    return output_directory / filename


# Save the current figure
def _save_or_show(output_file=None, show_plot=False):
    if output_file is not None:
        plt.savefig(output_file, dpi=DPI, bbox_inches="tight")

    # Show plots for interactive use if requested
    if show_plot:
        plt.show()

    plt.close()


# Build the output filename using an optional prefix
def _build_filename(name, prefix=None, file_format="png"):
    if prefix is None or prefix == "":
        return f"{name}.{file_format}"

    return f"{name}_{prefix}.{file_format}"


# Formatting for 2D plots
def _format_2d_axes(ax, show_legend=True):
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)

    if GRID:
        ax.grid(True)

    if show_legend:
        ax.legend(fontsize=LEGEND_SIZE)


# Formatting for 3D plots
def _format_3d_axes(ax, show_legend=True):
    ax.tick_params(axis="x", labelsize=TICK_LABEL_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_SIZE)
    ax.tick_params(axis="z", labelsize=TICK_LABEL_SIZE)

    if show_legend:
        ax.legend(fontsize=LEGEND_SIZE)


# Plot the 3D trajectories
def plot_trajectory(results, output_file=None, show_plot=False, language=DEFAULT_LANGUAGE, show_legend=True):

    text = get_text(language)
    results_list = _as_list(results)

    fig = plt.figure(figsize=FIGURE_SIZE_3D)
    ax = fig.add_subplot(111, projection="3d")

    # Use the longest target history because each law may stop at a different time
    reference_result = max(results_list, key=lambda item: len(item["time"]))
    rT = reference_result["rT"]

    # Target trajectory is plotted first to ensure it appears behind the pursuer trajectories
    ax.plot(
        rT[:, 0],
        rT[:, 1],
        rT[:, 2],
        color=TARGET_COLOR,
        linewidth=LINE_WIDTH,
        label=text["target"],
    )

    ax.scatter(
        rT[0, 0],
        rT[0, 1],
        rT[0, 2],
        marker="o",
        color=TARGET_COLOR,
        s=MARKER_MEDIUM,
        label=text["target_start"],
    )

    ax.scatter(
        rT[-1, 0],
        rT[-1, 1],
        rT[-1, 2],
        marker="o",
        facecolors="none",
        edgecolors=TARGET_COLOR,
        s=MARKER_SMALL,
        label=text["target_end"],
        depthshade=False,
    )

    # Plot the pursuer trajectory(ies)
    for result in results_list:
        rM = result["rM"]
        label = _guidance_label(result)
        color = _guidance_color(result)

        ax.plot(
            rM[:, 0],
            rM[:, 1],
            rM[:, 2],
            color=color,
            linewidth=LINE_WIDTH,
            label=f"{text['pursuer']} - {label}",
        )

        ax.scatter(
            rM[0, 0],
            rM[0, 1],
            rM[0, 2],
            color=color,
            s=MARKER_MEDIUM,
            marker="o",
        )

        # Mark the interception point if capture occurred; otherwise mark the final point
        intercepted = bool(result.get("intercepted", False))
        time = np.asarray(result.get("time", []), dtype=float)
        R = np.asarray(result.get("R", []), dtype=float)
        capture_radius = result.get("capture_radius", None)
        intercept_time = result.get("intercept_time", None)

        if intercepted:
            if intercept_time is not None and time.size > 0:
                marker_index = int(np.argmin(np.abs(time - intercept_time)))
            elif capture_radius is not None and R.size > 0:
                capture_indices = np.where(R <= capture_radius)[0]
                marker_index = int(capture_indices[0]) if capture_indices.size > 0 else -1
            else:
                marker_index = -1

            ax.scatter(
                rM[marker_index, 0],
                rM[marker_index, 1],
                rM[marker_index, 2],
                color=color,
                s=MARKER_BIG,
                marker="x",
            )

        else:
            ax.scatter(
                rM[-1, 0],
                rM[-1, 1],
                rM[-1, 2],
                marker="o",
                facecolors="none",
                edgecolors=color,
                s=MARKER_SMALL,
                depthshade=False,
            )

    ax.set_xlabel(text["x"], fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel(text["y"], fontsize=AXIS_LABEL_SIZE)
    ax.set_zlabel(text["z"], fontsize=AXIS_LABEL_SIZE)

    if USE_FIGURE_TITLES:
        ax.set_title(text["trajectory_title"])

    _format_3d_axes(ax, show_legend=show_legend)

    _save_or_show(output_file, show_plot)


# Plot relative distance
def plot_range(results, output_file=None, show_plot=False, language=DEFAULT_LANGUAGE, show_legend=True):

    text = get_text(language)
    results_list = _as_list(results)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_2D)

    capture_radius = results_list[0].get("capture_radius", None)

    for result in results_list:
        time = result["time"]
        R = result["R"]
        label = _guidance_label(result)
        color = _guidance_color(result)

        ax.plot(
            time,
            R,
            label=label,
            color=color,
            linewidth=LINE_WIDTH,
        )

        time_at_min_distance = result.get("time_at_min_distance", None)
        min_distance = result.get("min_distance", None)

        # Mark the closest approach only when it satisfies the capture condition
        captured = (
            capture_radius is not None
            and min_distance is not None
            and min_distance <= capture_radius
        )

        if captured and time_at_min_distance is not None:
            ax.scatter(
                [time_at_min_distance],
                [min_distance],
                marker="x",
                s=MARKER_BIG,
                label=f"{label} - {text['minimum_distance']}",
                color=color,
            )

    if capture_radius is not None:
        ax.axhline(
            capture_radius,
            color=LIMIT_COLOR,
            linestyle=LIMIT_LINE,
            linewidth=LIMIT_LINE_WIDTH,
            label=text["capture_radius"],
        )

    ax.set_xlabel(text["time"], fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel(text["relative_distance"], fontsize=AXIS_LABEL_SIZE)

    if USE_FIGURE_TITLES:
        ax.set_title(text["range_title"])

    _format_2d_axes(ax, show_legend=show_legend)

    _save_or_show(output_file, show_plot)


# Plot acceleration magnitudes
def plot_acceleration(
    results,
    output_file=None,
    show_plot=False,
    language=DEFAULT_LANGUAGE,
    show_legend=True,
):
    text = get_text(language)
    results_list = _as_list(results)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_2D)

    # Applied acceleration is plotted only when lateral saturation is active
    plot_applied = any(
        result.get("acceleration_saturation", False)
        and result.get("a_cmd_max", None) is not None
        and not np.isinf(result.get("a_cmd_max", np.inf))
        for result in results_list
    )

    # Total acceleration is plotted only when longitudinal acceleration exists
    plot_total = _has_nonzero_history(results_list, "aM_parallel")

    for result in results_list:
        time = result["time"]
        label = _guidance_label(result)
        color = _guidance_color(result)

        a_cmd = result["a_cmd"]
        a_applied = result.get("a_applied", a_cmd)
        aM_total = result.get("aM_total", None)

        a_cmd_norm = _norm_history(a_cmd)

        # Plot spikes are hidden for better visualization
        if REMOVE_PLOT_SPIKES:
            a_cmd_norm = _remove_plot_spikes(
                a_cmd_norm,
                reference_limit=PLOT_SPIKE_REFERENCE,
                factor=PLOT_SPIKE_FACTOR,
            )

        n_cmd = min(len(time), len(a_cmd_norm))

        if plot_applied or plot_total:
            commanded_label = f"{label} - {text['commanded']}"
        else:
            commanded_label = label

        ax.plot(
            time[:n_cmd],
            a_cmd_norm[:n_cmd],
            color=color,
            linestyle="-",
            linewidth=LINE_WIDTH,
            label=commanded_label,
        )

        if plot_applied:
            a_applied_norm = _norm_history(a_applied)
            n_applied = min(len(time), len(a_applied_norm))

            ax.plot(
                time[:n_applied],
                a_applied_norm[:n_applied],
                color=color,
                linestyle="--",
                linewidth=LINE_WIDTH,
                label=f"{label} - {text['applied']}",
            )

        if plot_total and aM_total is not None:
            aM_total_norm = _norm_history(aM_total)

            if REMOVE_PLOT_SPIKES:
                aM_total_norm = _remove_plot_spikes(
                    aM_total_norm,
                    reference_limit=PLOT_SPIKE_REFERENCE,
                    factor=PLOT_SPIKE_FACTOR,
                )

            n_total = min(len(time), len(aM_total_norm))

            ax.plot(
                time[:n_total],
                aM_total_norm[:n_total],
                color=color,
                linestyle=":",
                linewidth=LINE_WIDTH,
                label=f"{label} - {text['total']}",
            )

    a_cmd_max = results_list[0].get("a_cmd_max", None)
    use_saturation = results_list[0].get("acceleration_saturation", False)

    if use_saturation and a_cmd_max is not None and not np.isinf(a_cmd_max):
        ax.axhline(
            a_cmd_max,
            color=LIMIT_COLOR,
            linestyle=LIMIT_LINE,
            linewidth=LIMIT_LINE_WIDTH,
            label=text["guidance_command_limit"],
        )

    ax.set_xlabel(text["time"], fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel(text["acceleration_magnitude"], fontsize=AXIS_LABEL_SIZE)

    if USE_FIGURE_TITLES:
        ax.set_title(text["acceleration_title"])

    _format_2d_axes(ax, show_legend=show_legend)

    _save_or_show(output_file, show_plot)


# Plot the prescribed longitudinal acceleration history
def plot_longitudinal_acceleration(results, output_file=None, show_plot=False, language=DEFAULT_LANGUAGE, show_legend=True):

    text = get_text(language)
    results_list = _as_list(results)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_2D)

    for result in results_list:
        if "aM_parallel" not in result:
            continue

        time = result["time"]
        label = _guidance_label(result)
        color = _guidance_color(result)

        # Positive values accelerate the pursuer; negative values decelerate it
        aM_parallel = np.asarray(result["aM_parallel"], dtype=float)
        n_parallel = min(len(time), len(aM_parallel))

        ax.plot(
            time[:n_parallel],
            aM_parallel[:n_parallel],
            color=color,
            linewidth=LINE_WIDTH,
            label=label,
        )

    ax.set_xlabel(text["time"], fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel(text["longitudinal_acceleration"], fontsize=AXIS_LABEL_SIZE)

    if USE_FIGURE_TITLES:
        ax.set_title(text["longitudinal_acceleration_title"])

    _format_2d_axes(ax, show_legend=show_legend)

    _save_or_show(output_file, show_plot)


# Plot the closing velocity history
def plot_closing_velocity(results, output_file=None, show_plot=False, language=DEFAULT_LANGUAGE, show_legend=True):

    text = get_text(language)
    results_list = _as_list(results)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_2D)

    for result in results_list:
        time = result["time"]
        Vc = result["Vc"]
        label = _guidance_label(result)
        color = _guidance_color(result)

        n_velocity = min(len(time), len(Vc))

        ax.plot(
            time[:n_velocity],
            Vc[:n_velocity],
            color=color,
            linewidth=LINE_WIDTH,
            label=label,
        )

    ax.set_xlabel(text["time"], fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel(text["closing_velocity"], fontsize=AXIS_LABEL_SIZE)

    if USE_FIGURE_TITLES:
        ax.set_title(text["closing_velocity_title"])

    _format_2d_axes(ax, show_legend=show_legend)

    _save_or_show(output_file, show_plot)


# Plot the pursuer speed history
def plot_pursuer_speed(results, output_file=None, show_plot=False, language=DEFAULT_LANGUAGE, show_legend=True):

    text = get_text(language)
    results_list = _as_list(results)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_2D)

    for result in results_list:
        time = result["time"]
        color = _guidance_color(result)

        if "VM" in result:
            VM = result["VM"]
        else:
            VM = _norm_history(result["vM"])

        label = _guidance_label(result)
        n_speed = min(len(time), len(VM))

        ax.plot(
            time[:n_speed],
            VM[:n_speed],
            color=color,
            linewidth=LINE_WIDTH,
            label=label,
        )

    ax.set_xlabel(text["time"], fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel(text["pursuer_speed"], fontsize=AXIS_LABEL_SIZE)

    if USE_FIGURE_TITLES:
        ax.set_title(text["pursuer_speed_title"])

    _format_2d_axes(ax, show_legend=show_legend)

    _save_or_show(output_file, show_plot)


# Generate the standard plots for one scenario or a group of laws
def plot_results(
    results,
    output_directory=None,
    output_dir=None,
    prefix="results",
    file_format="png",
    show_plots=False,
    include_closing_velocity=False,
    language=DEFAULT_LANGUAGE,
    show_legend=True,
):
    if output_directory is None:
        output_directory = output_dir

    show_plot = show_plots if output_directory is not None else True

    trajectory_file = _make_output_file(
        output_directory,
        _build_filename("trajectory", prefix, file_format),
    )

    range_file = _make_output_file(
        output_directory,
        _build_filename("range", prefix, file_format),
    )

    acceleration_file = _make_output_file(
        output_directory,
        _build_filename("acceleration", prefix, file_format),
    )

    pursuer_speed_file = _make_output_file(
        output_directory,
        _build_filename("pursuer_speed", prefix, file_format),
    )

    plot_trajectory(
        results,
        output_file=trajectory_file,
        show_plot=show_plot,
        language=language,
        show_legend=show_legend,
    )

    plot_range(
        results,
        output_file=range_file,
        show_plot=show_plot,
        language=language,
        show_legend=show_legend,
    )

    plot_acceleration(
        results,
        output_file=acceleration_file,
        show_plot=show_plot,
        language=language,
        show_legend=show_legend,
    )

    plot_pursuer_speed(
        results,
        output_file=pursuer_speed_file,
        show_plot=show_plot,
        language=language,
        show_legend=show_legend,
    )

    # Hot fix: the closing velocity plot is optional because it is not always useful
    if include_closing_velocity:
        closing_velocity_file = _make_output_file(
            output_directory,
            _build_filename("closing_velocity", prefix, file_format),
        )

        plot_closing_velocity(
            results,
            output_file=closing_velocity_file,
            show_plot=show_plot,
            language=language,
            show_legend=show_legend,
        )