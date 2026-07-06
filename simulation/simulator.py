import numpy as np
from guidance.guidance_laws import pure_pursuit, true_proportional_navigation, pure_proportional_navigation, augmented_proportional_navigation, relative_quantities
from utilities.functions import unit, norm, saturate


# Simulate one pursuer-target engagement
def simulate_engagement(scenario=None):

    if scenario is None:
        raise ValueError("A scenario must be provided to simulate_engagement().")

    # Simulation settings
    guidance_law = scenario.get("guidance_law", "TPN").upper()
    dt = scenario["dt"]
    t_final = scenario["t_final"]
    capture_radius = scenario["capture_radius"]
    verbose = scenario.get("verbose", False)

    # Stopping criteria
    stop_at_capture = scenario.get("stop_at_capture", False)
    stop_at_closest_approach = scenario.get("stop_at_closest_approach", True)
    closest_approach_confirm_steps = scenario.get("closest_approach_confirm_steps", 1)

    # Guidance gains
    N = scenario["N"]
    k = scenario["k"]

    # Guidance command saturation
    use_acceleration_saturation = scenario.get("acceleration_saturation", False)
    a_cmd_max = scenario.get("a_cmd_max", np.inf)

    # Discrete guidance update
    use_discrete_guidance = scenario.get("discrete_guidance", False)
    guidance_update_interval = scenario.get("guidance_update_interval", dt)

    if use_discrete_guidance:
        guidance_update_steps = max(1, int(round(guidance_update_interval / dt)))
    else:
        guidance_update_steps = 1

    actual_guidance_update_interval = guidance_update_steps * dt

    # Prescribed longitudinal acceleration
    use_variable_acceleration = scenario.get("variable_acceleration", False)
    pursuer_acceleration_function = scenario.get("pursuer_acceleration_function", None)

    # Initial states
    rM = scenario["rM0"].copy()
    vM = scenario["vM0"].copy()
    rT = scenario["rT0"].copy()
    vT = scenario["vT0"].copy()

    # Constant pursuer speed used when longitudinal acceleration is disabled
    V_M_constant = norm(vM)

    # State histories
    time_history = []
    rM_history = []
    vM_history = []
    rT_history = []
    vT_history = []

    # Relative quantity histories
    R_history = []
    Vc_history = []
    omega_los_history = []

    # Acceleration and speed histories
    aT_history = []
    VM_history = []
    a_cmd_history = []
    a_applied_history = []
    saturation_history = []
    guidance_update_history = []
    a_guidance_limit_history = []
    aM_parallel_history = []
    aM_longitudinal_history = []
    aM_total_history = []

    # Current guidance commands
    a_cmd_current = np.zeros(3)
    a_applied_current = np.zeros(3)

    # Engagement outcome
    intercepted = False
    intercept_time = None

    min_distance = np.inf
    time_at_min_distance = None

    closest_approach_reached = False
    closest_approach_time = None
    increasing_distance_count = 0

    number_of_steps = int(t_final / dt)

    for i in range(number_of_steps + 1):
        t = i * dt

        # Target acceleration
        if callable(scenario.get("target_acceleration_function", None)):
            aT = np.asarray(scenario["target_acceleration_function"](t), dtype=float)
        else:
            aT = np.asarray(scenario.get("aT0", np.zeros(3)), dtype=float)

        # Relative geometry used by the guidance laws
        r_rel, v_rel, R, lambda_hat, Vc, omega_los = relative_quantities(rM, vM, rT, vT)

        # Store state data
        time_history.append(t)
        rM_history.append(rM.copy())
        vM_history.append(vM.copy())
        rT_history.append(rT.copy())
        vT_history.append(vT.copy())

        # Store relative quantities
        R_history.append(R)
        Vc_history.append(Vc)
        omega_los_history.append(omega_los.copy())

        # Store target acceleration and pursuer speed
        aT_history.append(aT.copy())
        VM_history.append(norm(vM))

        # Update minimum distance
        if R < min_distance:
            min_distance = R
            time_at_min_distance = t

        # Register capture without necessarily stopping the simulation
        if R <= capture_radius and not intercepted:
            intercepted = True
            intercept_time = t

            if verbose:
                print(f"Capture radius reached at t = {t:.2f} s")

        # Update guidance command according to the selected law
        update_guidance = (i % guidance_update_steps == 0)

        if update_guidance:
            if guidance_law == "PP":
                a_cmd_current = pure_pursuit(rM, vM, rT, k)

            elif guidance_law == "TPN":
                a_cmd_current = true_proportional_navigation(rM, vM, rT, vT, N)

            elif guidance_law == "PPN":
                a_cmd_current = pure_proportional_navigation(rM, vM, rT, vT, N)

            elif guidance_law == "APN":
                a_cmd_current = augmented_proportional_navigation(rM, vM, rT, vT, aT, N)

            else:
                raise ValueError("Invalid guidance law. Use 'PP', 'TPN', 'PPN' or 'APN'.")

        # Apply saturation only to the lateral guidance command
        if use_acceleration_saturation and not np.isinf(a_cmd_max):
            a_guidance_limit = a_cmd_max
            a_applied_current = saturate(a_cmd_current, a_guidance_limit)
        else:
            a_guidance_limit = np.inf
            a_applied_current = a_cmd_current.copy()

        saturated = norm(a_applied_current - a_cmd_current) > 1e-9

        # Longitudinal acceleration prescribed along the pursuer velocity direction
        if use_variable_acceleration and callable(pursuer_acceleration_function):
            aM_parallel = float(pursuer_acceleration_function(t))
        else:
            aM_parallel = 0.0

        vM_hat = unit(vM)
        aM_longitudinal = aM_parallel * vM_hat

        # Total pursuer acceleration
        aM_total = a_applied_current + aM_longitudinal

        # Store acceleration data
        a_cmd_history.append(a_cmd_current.copy())
        a_applied_history.append(a_applied_current.copy())
        saturation_history.append(saturated)
        guidance_update_history.append(update_guidance)
        a_guidance_limit_history.append(a_guidance_limit)
        aM_parallel_history.append(aM_parallel)
        aM_longitudinal_history.append(aM_longitudinal.copy())
        aM_total_history.append(aM_total.copy())

        # Stop after capture if requested, or continue until closest approach
        if intercepted:

            if stop_at_capture:
                if verbose:
                    print(f"Simulation stopped at capture radius, t = {t:.2f} s")
                break

            if stop_at_closest_approach:

                if len(R_history) >= 2 and R_history[-1] > R_history[-2]:
                    increasing_distance_count += 1
                else:
                    increasing_distance_count = 0

                if Vc <= 0.0 or increasing_distance_count >= closest_approach_confirm_steps:
                    closest_approach_reached = True
                    closest_approach_time = time_at_min_distance

                    if verbose:
                        print(
                            f"Closest approach reached at t = {time_at_min_distance:.2f} s "
                            f"with R_min = {min_distance:.3f} m"
                        )

                    break

        # Update pursuer velocity
        if use_variable_acceleration:
            vM = vM + aM_total * dt

        else:
            # Without longitudinal acceleration, only the velocity direction changes
            vM_temp = vM + a_applied_current * dt

            if norm(vM_temp) > 1e-12:
                vM = V_M_constant * unit(vM_temp)

        # Update target velocity and positions
        vT = vT + aT * dt
        rM = rM + vM * dt
        rT = rT + vT * dt

    # Return all histories and final metrics needed for post-processing
    results = {
        "scenario_name": scenario["name"],
        "guidance_law": guidance_law,
        "dt": dt,
        "t_final": t_final,
        "N": N,
        "k": k,
        "capture_radius": capture_radius,
        "verbose": verbose,
        "stop_at_capture": stop_at_capture,
        "stop_at_closest_approach": stop_at_closest_approach,
        "closest_approach_confirm_steps": closest_approach_confirm_steps,
        "acceleration_saturation": use_acceleration_saturation,
        "a_cmd_max": a_cmd_max,
        "discrete_guidance": use_discrete_guidance,
        "guidance_update_interval": guidance_update_interval,
        "guidance_update_steps": guidance_update_steps,
        "actual_guidance_update_interval": actual_guidance_update_interval,
        "variable_acceleration": use_variable_acceleration,
        "time": np.array(time_history),
        "rM": np.array(rM_history),
        "vM": np.array(vM_history),
        "rT": np.array(rT_history),
        "vT": np.array(vT_history),
        "R": np.array(R_history),
        "Vc": np.array(Vc_history),
        "omega_los": np.array(omega_los_history),
        "aT": np.array(aT_history),
        "VM": np.array(VM_history),
        "a_cmd": np.array(a_cmd_history),
        "a_applied": np.array(a_applied_history),
        "saturated": np.array(saturation_history),
        "guidance_updated": np.array(guidance_update_history),
        "a_guidance_limit": np.array(a_guidance_limit_history),
        "aM_parallel": np.array(aM_parallel_history),
        "aM_longitudinal": np.array(aM_longitudinal_history),
        "aM_total": np.array(aM_total_history),
        "intercepted": intercepted,
        "intercept_time": intercept_time,
        "min_distance": min_distance,
        "time_at_min_distance": time_at_min_distance,
        "closest_approach_reached": closest_approach_reached,
        "closest_approach_time": closest_approach_time,
    }

    return results