import numpy as np
import pybullet as p
class CPG:
    def __init__(self, frequency=1.0, amplitude=0.3, duty_factor=0.75):
        self.frequency = frequency  # Hz
        self.amplitude = amplitude  # Amplitud de la oscilación
        self.duty_factor = duty_factor  # Proporción entre fase de apoyo y fase de balanceo
        self.phase = np.zeros(4)  # 4 fases para las 4 patas del robot
    
    def update(self, delta_t):
        # Actualiza la fase de cada pata según la frecuencia
        self.phase += 2 * np.pi * self.frequency * delta_t
        self.phase %= 2 * np.pi  # Mantener las fases dentro de [0, 2*pi]
        
    def get_motor_angles(self):
        # Calcula las posiciones de los motores de las patas usando la fase
        angles = []
        for i in range(4):
            # Genera las posiciones de las patas basadas en las fases
            phase_shift = (i * np.pi / 2)  # Fases desplazadas para las patas
            angle = self.amplitude * np.cos(self.phase[i] + phase_shift)
            angles.append(angle)
        return np.array(angles)
