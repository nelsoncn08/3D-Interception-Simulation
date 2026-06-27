DEFAULT_LANGUAGE = "pt"

USE_FIGURE_TITLES = False


TEXTS = {
    "pt": {
        "time": "Tempo [s]",
        "relative_distance": "Distância relativa [m]",
        "acceleration_magnitude": "Magnitude da aceleração [m/s²]",
        "longitudinal_acceleration": "Aceleração longitudinal [m/s²]",
        "closing_velocity": "Velocidade de fechamento [m/s]",
        "pursuer_speed": "Velocidade do perseguidor [m/s]",

        "x": "x [m]",
        "y": "y [m]",
        "z": "z [m]",

        "target": "Alvo",
        "target_start": "Alvo - início",
        "target_end": "Alvo - fim",
        "pursuer": "Perseguidor",

        "capture_radius": "Raio de captura",
        "guidance_command_limit": "Limite do comando de guiagem",
        "minimum_distance": "$R_{min}$",
        "zero_longitudinal_acceleration": "Aceleração longitudinal nula",
        "zero_closing_velocity": "Velocidade de fechamento nula",

        "commanded": "comandada",
        "applied": "aplicada",
        "total": "total",

        "trajectory_title": "Engajamento perseguidor-alvo",
        "range_title": "Histórico da distância relativa",
        "acceleration_title": "Histórico das acelerações",
        "longitudinal_acceleration_title": "Histórico da aceleração longitudinal",
        "closing_velocity_title": "Histórico da velocidade de fechamento",
        "pursuer_speed_title": "Histórico da velocidade do perseguidor",
    },

    "en": {
        "time": "Time [s]",
        "relative_distance": "Relative distance [m]",
        "acceleration_magnitude": "Acceleration magnitude [m/s²]",
        "longitudinal_acceleration": "Longitudinal acceleration [m/s²]",
        "closing_velocity": "Closing velocity [m/s]",
        "pursuer_speed": "Pursuer speed [m/s]",

        "x": "x [m]",
        "y": "y [m]",
        "z": "z [m]",

        "target": "Target",
        "target_start": "Target start",
        "target_end": "Target end",
        "pursuer": "Pursuer",

        "capture_radius": "Capture radius",
        "guidance_command_limit": "Guidance command limit",
        "minimum_distance": "$R_{min}$",
        "zero_longitudinal_acceleration": "Zero longitudinal acceleration",
        "zero_closing_velocity": "Zero closing velocity",

        "commanded": "commanded",
        "applied": "applied",
        "total": "total",

        "trajectory_title": "Pursuer-target engagement",
        "range_title": "Relative distance history",
        "acceleration_title": "Guidance and total acceleration history",
        "longitudinal_acceleration_title": "Longitudinal acceleration history",
        "closing_velocity_title": "Closing velocity history",
        "pursuer_speed_title": "Pursuer speed history",
    },
}


def get_text(language=DEFAULT_LANGUAGE):
    if language not in TEXTS:
        raise ValueError("Invalid language. Use 'pt' or 'en'.")

    return TEXTS[language]