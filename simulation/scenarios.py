import numpy as np

STANDARD_GRAVITY = 9.80665
GUIDANCE_ACCELERATION_LIMIT_G = 0.5
GUIDANCE_ACCELERATION_LIMIT = GUIDANCE_ACCELERATION_LIMIT_G * STANDARD_GRAVITY

GUIDANCE_UPDATE_INTERVAL = 0.1


# Base parameters shared by all deterministic scenarios
def base_scenario(guidance_law="TPN"):
    scenario = {
        "name": "Base interception",
        "guidance_law": guidance_law,

        "dt": 0.01,
        "t_final": 400.0,
        "capture_radius": 5.0,

        # Capture is registered, but the simulation may continue until closest approach
        "stop_at_capture": False,
        "stop_at_closest_approach": True,
        "closest_approach_confirm_steps": 1,
        "verbose": False,

        # Guidance gains
        "N": 3.0,
        "k": 0.75,

        "rM0": np.array([0.0, 0.0, 0.0]),

        # Constant speed for cases without longitudinal acceleration
        "vM0": np.array([543.8, 0.0, 0.0]),

        # Initial target state
        "rT0": np.array([48000.0, 12000.0, 4000.0]),  # Approximately 50 km
        "vT0": np.array([-50.0, 0.0, -10.0]),         # Approximately 51 m/s
        "aT0": np.array([0.0, 0.0, 0.0]),

        # Default ideal-case settings
        "a_cmd_max": np.inf,
        "guidance_update_interval": 0.01,

        "target_maneuver": False,
        "acceleration_saturation": False,
        "discrete_guidance": False,
        "variable_acceleration": False,
        "pursuer_acceleration_function": None,
    }

    return scenario


# Nominal scenario with non-accelerating target
def nominal(guidance_law="TPN"):
    scenario = base_scenario(guidance_law)

    scenario["name"] = "Nominal interception"
    scenario["target_maneuver"] = False
    scenario["aT0"] = np.array([0.0, 0.0, 0.0])

    return scenario


# Scenario with maneuvering target
def maneuvering_target(guidance_law="TPN"):
    scenario = base_scenario(guidance_law)

    scenario["name"] = "Target lateral acceleration interception"
    scenario["target_maneuver"] = True

    # Mild lateral acceleration used for a Shahed-like target
    scenario["aT0"] = np.array([0.0, 1.0, 0.0])

    return scenario


# Scenario with lateral guidance command saturation
def acceleration_saturation(guidance_law="TPN"):
    scenario = maneuvering_target(guidance_law)

    scenario["name"] = "Guidance command saturation interception"
    scenario["acceleration_saturation"] = True

    scenario["a_cmd_max"] = GUIDANCE_ACCELERATION_LIMIT

    return scenario


# Scenario with discrete guidance command updates
def discrete_guidance(guidance_law="TPN"):
    scenario = maneuvering_target(guidance_law)

    scenario["name"] = "Discrete guidance update interception"
    scenario["discrete_guidance"] = True

    scenario["guidance_update_interval"] = GUIDANCE_UPDATE_INTERVAL

    return scenario


# Scenario with variable acceleration of the pursuer
def variable_acceleration(guidance_law="TPN"):
    scenario = maneuvering_target(guidance_law)

    scenario["name"] = "Prescribed longitudinal acceleration interception"
    scenario["variable_acceleration"] = True

    # Hot fix: small initial speed only to define the velocity direction
    scenario["vM0"] = np.array([0.1, 0.0, 0.0])

    # Prescribed longitudinal acceleration profile [m/s²]
    def pursuer_acceleration_function(t):
        if t <= 4.0:
            return 45.0
        elif t <= 40.0:
            return 15.0
        else:
            return -1.0

    scenario["pursuer_acceleration_function"] = pursuer_acceleration_function

    return scenario


# Complete scenario combining all implementation effects
def complete(guidance_law="TPN"):
    scenario = variable_acceleration(guidance_law)

    scenario["name"] = "Complete interception scenario"

    scenario["acceleration_saturation"] = True
    scenario["a_cmd_max"] = GUIDANCE_ACCELERATION_LIMIT

    scenario["discrete_guidance"] = True
    scenario["guidance_update_interval"] = GUIDANCE_UPDATE_INTERVAL

    return scenario


SCENARIO_REGISTRY = {
    "nominal": nominal,
    "maneuvering_target": maneuvering_target,
    "acceleration_saturation": acceleration_saturation,
    "discrete_guidance": discrete_guidance,
    "variable_acceleration": variable_acceleration,
    "complete": complete,
}