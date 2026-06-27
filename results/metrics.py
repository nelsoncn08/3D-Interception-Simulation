import numpy as np


def _norm_history(values):
    values = np.asarray(values, dtype=float)

    if values.size == 0:
        return np.array([])

    if values.ndim == 1:
        return np.abs(values)

    return np.linalg.norm(values, axis=1)


def _effort(acceleration_norm, dt):
    acceleration_norm = np.asarray(acceleration_norm, dtype=float)

    if acceleration_norm.size == 0:
        return 0.0

    return float(np.sum(acceleration_norm**2) * dt)


def _max_or_zero(values):
    values = np.asarray(values, dtype=float)

    if values.size == 0:
        return 0.0

    return float(np.max(values))


def _min_or_zero(values):
    values = np.asarray(values, dtype=float)

    if values.size == 0:
        return 0.0

    return float(np.min(values))


def _format_value(value, precision=3):
    if value is None:
        return "-"

    if np.isinf(value):
        return "inf"

    return f"{value:.{precision}f}"


def _format_time(value, precision=3):
    if value is None:
        return "-"

    return f"{value:.{precision}f}"


def compute_metrics(results):
    time = np.asarray(results["time"], dtype=float)
    R = np.asarray(results["R"], dtype=float)

    # Command requested by the guidance law before saturation
    a_cmd = np.asarray(results["a_cmd"], dtype=float)

    # Lateral guidance acceleration effectively applied after saturation
    a_applied = np.asarray(results.get("a_applied", a_cmd), dtype=float)

    # Total pursuer acceleration: lateral guidance + longitudinal acceleration
    aM_total = np.asarray(results.get("aM_total", a_applied), dtype=float)

    # Signed longitudinal acceleration:
    # positive values accelerate the pursuer along its velocity direction;
    # negative values represent deceleration.
    aM_parallel = np.asarray(
        results.get("aM_parallel", np.zeros(len(time))),
        dtype=float
    )

    VM = np.asarray(results.get("VM", np.zeros(len(time))), dtype=float)
    saturated = np.asarray(results.get("saturated", np.zeros(len(time))), dtype=bool)

    capture_radius = results.get("capture_radius", None)

    if len(time) > 1:
        dt = float(results.get("dt", time[1] - time[0]))
    else:
        dt = float(results.get("dt", 0.0))

    # The capture event and the closest approach are not necessarily the same.
    # The simulation may register capture and continue until the minimum distance.
    if len(R) > 0:
        index_min_distance = int(np.argmin(R))
        min_distance = float(R[index_min_distance])
        time_at_min_distance = float(time[index_min_distance])
        final_distance = float(R[-1])
    else:
        min_distance = np.inf
        time_at_min_distance = None
        final_distance = np.inf

    final_time = float(time[-1]) if len(time) > 0 else 0.0

    intercepted = bool(results.get("intercepted", False))
    intercept_time = results.get("intercept_time", None)

    if capture_radius is not None and len(R) > 0:
        capture_indices = np.where(R <= capture_radius)[0]

        if len(capture_indices) > 0:
            intercepted = True

            if intercept_time is None:
                intercept_time = float(time[capture_indices[0]])

    if intercept_time is not None:
        intercept_time = float(intercept_time)

    closest_approach_reached = bool(results.get("closest_approach_reached", False))
    closest_approach_time = results.get("closest_approach_time", time_at_min_distance)

    if closest_approach_time is not None:
        closest_approach_time = float(closest_approach_time)

    # Norm histories used for maximum values and effort metrics
    a_cmd_norm = _norm_history(a_cmd)
    a_applied_norm = _norm_history(a_applied)
    aM_total_norm = _norm_history(aM_total)

    max_commanded_acceleration = _max_or_zero(a_cmd_norm)
    max_applied_acceleration = _max_or_zero(a_applied_norm)
    max_total_acceleration = _max_or_zero(aM_total_norm)

    # Effort metrics are computed as the time integral of acceleration squared
    commanded_control_effort = _effort(a_cmd_norm, dt)
    applied_control_effort = _effort(a_applied_norm, dt)
    total_acceleration_effort = _effort(aM_total_norm, dt)

    max_longitudinal_acceleration = _max_or_zero(aM_parallel)
    min_longitudinal_acceleration = _min_or_zero(aM_parallel)
    max_abs_longitudinal_acceleration = _max_or_zero(np.abs(aM_parallel))
    longitudinal_acceleration_effort = _effort(np.abs(aM_parallel), dt)

    # Saturation metrics indicate how often the requested command exceeded
    # the imposed lateral acceleration limit.
    saturation_count = int(np.sum(saturated))
    saturation_time = float(saturation_count * dt)

    if len(saturated) > 0:
        saturation_fraction = float(np.mean(saturated))
    else:
        saturation_fraction = 0.0

    a_guidance_limit = results.get("a_guidance_limit", None)

    if a_guidance_limit is not None:
        a_guidance_limit = np.asarray(a_guidance_limit, dtype=float)
        finite_limits = a_guidance_limit[np.isfinite(a_guidance_limit)]

        if len(finite_limits) > 0:
            max_guidance_limit = float(np.max(finite_limits))
        else:
            max_guidance_limit = np.inf
    else:
        max_guidance_limit = None

    if len(VM) > 0:
        max_pursuer_speed = float(np.max(VM))
        final_pursuer_speed = float(VM[-1])
    else:
        max_pursuer_speed = 0.0
        final_pursuer_speed = 0.0

    metrics = {
        "scenario_name": results.get("scenario_name", "Unknown"),
        "guidance_law": results.get("guidance_law", "Unknown"),

        "intercepted": intercepted,
        "intercept_time": intercept_time,
        "capture_radius": capture_radius,

        "min_distance": min_distance,
        "time_at_min_distance": time_at_min_distance,
        "closest_approach_reached": closest_approach_reached,
        "closest_approach_time": closest_approach_time,

        "final_distance": final_distance,
        "final_time": final_time,

        "max_commanded_acceleration": max_commanded_acceleration,
        "max_applied_acceleration": max_applied_acceleration,
        "max_total_acceleration": max_total_acceleration,

        "commanded_control_effort": commanded_control_effort,
        "applied_control_effort": applied_control_effort,
        "total_acceleration_effort": total_acceleration_effort,

        "max_longitudinal_acceleration": max_longitudinal_acceleration,
        "min_longitudinal_acceleration": min_longitudinal_acceleration,
        "max_abs_longitudinal_acceleration": max_abs_longitudinal_acceleration,
        "longitudinal_acceleration_effort": longitudinal_acceleration_effort,

        "max_guidance_limit": max_guidance_limit,

        "saturation_count": saturation_count,
        "saturation_time": saturation_time,
        "saturation_fraction": saturation_fraction,

        "max_pursuer_speed": max_pursuer_speed,
        "final_pursuer_speed": final_pursuer_speed,
    }

    return metrics


def format_metrics(metrics):
    intercepted_text = "yes" if metrics["intercepted"] else "no"
    closest_approach_text = "yes" if metrics["closest_approach_reached"] else "no"

    text = (
        "\nSimulation summary\n"
        "------------------\n"
        f"Scenario: {metrics['scenario_name']}\n"
        f"Guidance law: {metrics['guidance_law']}\n"
        f"Intercepted: {intercepted_text}\n"
        f"Intercept time: {_format_time(metrics['intercept_time'])} s\n"
        f"Minimum distance: {metrics['min_distance']:.3f} m\n"
        f"Time at minimum distance: {_format_time(metrics['time_at_min_distance'])} s\n"
        f"Closest approach reached: {closest_approach_text}\n"
        f"Final distance: {metrics['final_distance']:.3f} m\n"
        f"Final time: {metrics['final_time']:.3f} s\n"
        f"Maximum commanded acceleration: {metrics['max_commanded_acceleration']:.3f} m/s^2\n"
        f"Maximum applied acceleration: {metrics['max_applied_acceleration']:.3f} m/s^2\n"
        f"Applied guidance effort: {metrics['applied_control_effort']:.3f}\n"
        f"Saturation time: {metrics['saturation_time']:.3f} s\n"
        f"Saturation fraction: {100.0 * metrics['saturation_fraction']:.2f}%\n"
        f"Maximum pursuer speed: {metrics['max_pursuer_speed']:.3f} m/s\n"
        f"Final pursuer speed: {metrics['final_pursuer_speed']:.3f} m/s"
    )

    return text


def print_metrics(metrics):
    print(format_metrics(metrics))


def compare_metrics(metrics_list):
    print("\nMetrics comparison")
    print("------------------")
    print(
        f"{'Law':<8} "
        f"{'Success':>8} "
        f"{'t_int [s]':>12} "
        f"{'R_min [m]':>12} "
        f"{'t_Rmin [s]':>12} "
        f"{'a_cmd,max':>12} "
        f"{'a_app,max':>12} "
        f"{'J_app':>12} "
        f"{'Sat [%]':>10}"
    )

    for metrics in metrics_list:
        success = "yes" if metrics["intercepted"] else "no"

        print(
            f"{metrics['guidance_law']:<8} "
            f"{success:>8} "
            f"{_format_time(metrics['intercept_time']):>12} "
            f"{metrics['min_distance']:>12.3f} "
            f"{_format_time(metrics['time_at_min_distance']):>12} "
            f"{metrics['max_commanded_acceleration']:>12.3f} "
            f"{metrics['max_applied_acceleration']:>12.3f} "
            f"{metrics['applied_control_effort']:>12.3f} "
            f"{100.0 * metrics['saturation_fraction']:>10.2f}"
        )