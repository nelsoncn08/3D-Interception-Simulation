import numpy as np
from utilities.functions import norm, unit


# Relative quantities used by LOS-based guidance laws
def relative_quantities(rM, vM, rT, vT):
    rM = np.asarray(rM, dtype=float)
    vM = np.asarray(vM, dtype=float)
    rT = np.asarray(rT, dtype=float)
    vT = np.asarray(vT, dtype=float)

    r_rel = rT - rM
    v_rel = vT - vM
    R = norm(r_rel)

    # Avoid division by zero when the pursuer is too close to the target
    if R < 1e-9:
        lambda_hat = np.zeros(3)
        Vc = 0.0
        omega_los = np.zeros(3)

    else:
        lambda_hat = r_rel / R

        # Positive closing velocity means that the range is decreasing
        Vc = -np.dot(v_rel, lambda_hat)

        # The guidance command is disabled when the target is moving away
        if Vc < 0.0:
            Vc = 0.0

        omega_los = np.cross(r_rel, v_rel) / R**2

    return r_rel, v_rel, R, lambda_hat, Vc, omega_los


# Remove the acceleration component parallel to the pursuer velocity
def remove_parallel_component(a_cmd, vM):
    a_cmd = np.asarray(a_cmd, dtype=float)
    vM = np.asarray(vM, dtype=float)

    vM_hat = unit(vM)

    # If the velocity direction is undefined, keeps the original command
    if norm(vM_hat) < 1e-12:
        return a_cmd

    return a_cmd - np.dot(a_cmd, vM_hat) * vM_hat


# Pure pursuit (PP)
def pure_pursuit(rM, vM, rT, K):
    rM = np.asarray(rM, dtype=float)
    vM = np.asarray(vM, dtype=float)
    rT = np.asarray(rT, dtype=float)

    direction_to_target = rT - rM
    lambda_hat = unit(direction_to_target)

    VM = norm(vM)
    vM_hat = unit(vM)

    # If the velocity or LOS direction is undefined, returns zero command
    if VM < 1e-12 or norm(lambda_hat) < 1e-12:
        return np.zeros(3)

    # Direction error perpendicular to the pursuer velocity
    direction_error = lambda_hat - np.dot(lambda_hat, vM_hat) * vM_hat

    a_cmd = K * VM * direction_error

    # Keep only the lateral acceleration command
    a_cmd = remove_parallel_component(a_cmd, vM)

    return a_cmd


# True Proportional Navigation (TPN)
def true_proportional_navigation(rM, vM, rT, vT, N):
    r_rel, v_rel, R, lambda_hat, Vc, omega_los = relative_quantities(rM, vM, rT, vT)

    # Avoid command computation after interception or numerical singularity
    if R < 1e-9:
        return np.zeros(3)

    a_cmd = N * Vc * np.cross(omega_los, lambda_hat)

    # Keep only the lateral acceleration command
    a_cmd = remove_parallel_component(a_cmd, vM)

    return a_cmd


# Pure proportional navigation (PPN)
def pure_proportional_navigation(rM, vM, rT, vT, N):
    r_rel, v_rel, R, lambda_hat, Vc, omega_los = relative_quantities(rM, vM, rT, vT)

    VM = norm(vM)
    vM_hat = unit(vM)

    # Avoid command computation after interception or numerical singularity
    if R < 1e-9:
        return np.zeros(3)

    # In pure PN, the command is normal to the pursuer velocity
    a_cmd = N * VM * np.cross(omega_los, vM_hat)

    # Keep only the lateral acceleration command
    a_cmd = remove_parallel_component(a_cmd, vM)

    return a_cmd


# Augmented proportional navigation (APN)
def augmented_proportional_navigation(rM, vM, rT, vT, aT, N):
    aT = np.asarray(aT, dtype=float)

    r_rel, v_rel, R, lambda_hat, Vc, omega_los = relative_quantities(rM, vM, rT, vT)

    # Avoid command computation after interception or numerical singularity
    if R < 1e-9:
        return np.zeros(3)

    a_pn = N * Vc * np.cross(omega_los, lambda_hat)

    # Only the target acceleration perpendicular to the LOS contributes to APN
    aT_perp = aT - np.dot(aT, lambda_hat) * lambda_hat

    a_cmd = a_pn + (N / 2.0) * aT_perp

    # Keep only the lateral acceleration command
    a_cmd = remove_parallel_component(a_cmd, vM)

    return a_cmd