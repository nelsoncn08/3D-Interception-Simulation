import csv
import json
import re
from pathlib import Path

import numpy as np

from utilities.functions import norm


def sanitize_name(name):
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


def create_output_directory(results, root_directory="resultados"):
    scenario_name = sanitize_name(results["scenario_name"])
    guidance_law = sanitize_name(results["guidance_law"])

    output_directory = Path(root_directory) / scenario_name / guidance_law
    output_directory.mkdir(parents=True, exist_ok=True)

    return output_directory


def save_history_csv(results, output_directory):
    output_file = output_directory / "historico.csv"

    time = results["time"]
    R = results["R"]
    VM = results["VM"]
    Vc = results["Vc"]
    a_cmd = results["a_cmd"]
    a_applied = results["a_applied"]
    aM_parallel = results["aM_parallel"]
    aM_total = results["aM_total"]
    saturated = results["saturated"]
    guidance_updated = results["guidance_updated"]

    with open(output_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow([
            "time_s",
            "range_m",
            "pursuer_speed_m_s",
            "closing_velocity_m_s",
            "a_cmd_norm_m_s2",
            "a_applied_norm_m_s2",
            "aM_parallel_m_s2",
            "aM_total_norm_m_s2",
            "saturated",
            "guidance_updated",
        ])

        for i in range(len(time)):
            writer.writerow([
                time[i],
                R[i],
                VM[i],
                Vc[i],
                norm(a_cmd[i]),
                norm(a_applied[i]),
                aM_parallel[i],
                norm(aM_total[i]),
                bool(saturated[i]),
                bool(guidance_updated[i]),
            ])


def compute_summary(results):
    a_cmd_norm = np.linalg.norm(results["a_cmd"], axis=1)
    a_applied_norm = np.linalg.norm(results["a_applied"], axis=1)

    dt = results["dt"]
    saturated = results["saturated"]

    if len(saturated) > 0:
        saturation_time = float(np.sum(saturated) * dt)
        saturation_fraction = float(np.mean(saturated))
    else:
        saturation_time = 0.0
        saturation_fraction = 0.0

    control_effort = float(np.sum(a_applied_norm**2) * dt)

    summary = {
        "scenario_name": results["scenario_name"],
        "guidance_law": results["guidance_law"],
        "intercepted": bool(results["intercepted"]),
        "intercept_time_s": results["intercept_time"],
        "min_distance_m": float(results["min_distance"]),
        "time_at_min_distance_s": results["time_at_min_distance"],
        "closest_approach_reached": bool(results["closest_approach_reached"]),
        "closest_approach_time_s": results["closest_approach_time"],
        "max_commanded_acceleration_m_s2": float(np.max(a_cmd_norm)),
        "max_applied_acceleration_m_s2": float(np.max(a_applied_norm)),
        "control_effort_m2_s3": control_effort,
        "saturation_time_s": saturation_time,
        "saturation_fraction": saturation_fraction,
        "final_time_s": float(results["time"][-1]),
        "final_range_m": float(results["R"][-1]),
        "max_pursuer_speed_m_s": float(np.max(results["VM"])),
        "final_pursuer_speed_m_s": float(results["VM"][-1]),
    }

    return summary


def save_summary(results, output_directory):
    summary = compute_summary(results)

    json_file = output_directory / "resumo.json"
    txt_file = output_directory / "resumo.txt"

    with open(json_file, mode="w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4, ensure_ascii=False)

    with open(txt_file, mode="w", encoding="utf-8") as file:
        file.write(format_summary(summary))

    return summary


def format_summary(summary):
    intercepted_text = "sim" if summary["intercepted"] else "não"

    if summary["intercept_time_s"] is None:
        intercept_time_text = "--"
    else:
        intercept_time_text = f'{summary["intercept_time_s"]:.2f} s'

    text = (
        f'Cenário: {summary["scenario_name"]}\n'
        f'Lei de guiagem: {summary["guidance_law"]}\n'
        f'Interceptou: {intercepted_text}\n'
        f'Tempo de interceptação: {intercept_time_text}\n'
        f'Distância mínima: {summary["min_distance_m"]:.3f} m\n'
        f'Tempo na distância mínima: {summary["time_at_min_distance_s"]:.2f} s\n'
        f'Aceleração máxima comandada: {summary["max_commanded_acceleration_m_s2"]:.3f} m/s²\n'
        f'Aceleração máxima aplicada: {summary["max_applied_acceleration_m_s2"]:.3f} m/s²\n'
        f'Esforço de controle acumulado: {summary["control_effort_m2_s3"]:.3f}\n'
        f'Tempo em saturação: {summary["saturation_time_s"]:.3f} s\n'
        f'Fração em saturação: {100.0 * summary["saturation_fraction"]:.2f} %\n'
        f'Velocidade máxima do perseguidor: {summary["max_pursuer_speed_m_s"]:.3f} m/s\n'
        f'Velocidade final do perseguidor: {summary["final_pursuer_speed_m_s"]:.3f} m/s\n'
    )

    return text


def print_summary(summary):
    print("\n" + "=" * 60)
    print("RESUMO DA SIMULAÇÃO")
    print("=" * 60)
    print(format_summary(summary).strip())
    print("=" * 60 + "\n")


def save_all_outputs(results, root_directory="resultados"):
    output_directory = create_output_directory(results, root_directory)

    save_history_csv(results, output_directory)
    summary = save_summary(results, output_directory)

    return output_directory, summary