import numpy as np

def ik_leg_yz(y, z, L1=0.2522, L2=0.2510, debug=False):
    """
    Calcula los ángulos del hombro y rodilla para alcanzar la posición (y, z)
    en el plano YZ, relativa a la articulación de la cadera.
    - y: posición deseada del pie en el eje Y (adelante/atrás)
    - z: posición deseada del pie en el eje Z (vertical)
    Devuelve: (ángulo_hombro, ángulo_rodilla), o None si está fuera de alcance.
    Si debug=True, también calcula la cinemática directa para comprobar error.
    """
    # Comprobación de alcance
    D = (y**2 + z**2 - L1**2 - L2**2) / (2 * L1 * L2)
    if np.abs(D) > 1.0:
        if debug:
            print(f"⚠️ Posición fuera de alcance: (y={y:.3f}, z={z:.3f}), D={D:.3f}")
        return None

    # Resolver ángulos
    theta_knee = np.arccos(D)
    theta_hip = np.arctan2(z, y) - np.arctan2(L2 * np.sin(theta_knee), L1 + L2 * np.cos(theta_knee))
    # Ajuste de signo para rodilla
    theta_knee = -theta_knee

    if debug:
        # Cinemática directa para comprobación
        # Tener en cuenta signo invertido en rodilla
        th1 = theta_hip
        th2 = -theta_knee
        y_fk = L1 * np.cos(th1) + L2 * np.cos(th1 + th2)
        z_fk = L1 * np.sin(th1) + L2 * np.sin(th1 + th2)
        err_y = y_fk - y
        err_z = z_fk - z
        print(f"IK Debug: deseado (y={y:.3f}, z={z:.3f}), fk sim (y={y_fk:.3f}, z={z_fk:.3f}), err (dy={err_y:.4f}, dz={err_z:.4f})")

    return theta_hip, theta_knee
