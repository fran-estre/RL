
import numpy as np
import time
import pybullet as p
import pybullet_data
import os
from env import QuadrupedEnv
from ik_leg_yz import ik_leg_yz

# Carpeta para demos y logs
demo_dir = 'demos'
log_dir = os.path.join(demo_dir, 'logs')
os.makedirs(log_dir, exist_ok=True)

# 1) Inicializar entorno y obtener offsets neutrales
env = QuadrupedEnv()
obs, _ = env.reset()
initial_offsets = env.initial_joint_angles_rad  # neutrales

# 2) Dejar caer libremente sin motores activos
for _ in range(240):
    p.stepSimulation()
    time.sleep(1/240.0)

# 3) Medir stance real (local YZ)
leg_links = {
    "FR": ("FR_upper_leg", "toeFR"),
    "FL": ("FL_upper_leg", "toeFL"),
    "RR": ("RR_upper_leg", "toeRR"),
    "RL": ("RL_upper_leg", "toeRL")
}
stance_pos = {}
for leg, (hip_name, toe_name) in leg_links.items():
    idx_hip = next(i for i,n in env.link_name_map.items() if n == hip_name)
    idx_toe = next(i for i,n in env.link_name_map.items() if n == toe_name)
    hip_w = np.array(p.getLinkState(env.robot, idx_hip, computeForwardKinematics=True)[4])
    toe_w = np.array(p.getLinkState(env.robot, idx_toe, computeForwardKinematics=True)[4])
    stance_pos[leg] = [float(toe_w[1] - hip_w[1]), float(toe_w[2] - hip_w[2])]
print("Stance inicial:", stance_pos)

# 4) Parámetros de gait crawl
gait_order = ["FL","RR","FR","RL"]
joint_idxs = {"FR":[0,1,2],"FL":[3,4,5],"RR":[6,7,8],"RL":[9,10,11]}
advance_dy=0.03; swing_h=0.06; step_time=0.4; sim_steps=int(step_time/(1/240.0))

def bezier(p0,p1,p2,t): return (1-t)**2*np.array(p0)+2*(1-t)*t*np.array(p1)+t**2*np.array(p2)

# 5) Preparar log CSV
log_path = os.path.join(log_dir, 'foot_joint_log.csv')
with open(log_path, 'w') as f:
    header = ['step','leg','toe_y','toe_z','hip_angle','knee_angle']
    f.write(','.join(header) + '\n')

# 6) Generar demos y loguear en tiempo real
demo_obs, demo_act = [], []
step_count = 0
for cycle in range(2):
    for swing_leg in gait_order:
        print(f"🐾 Ciclo {cycle+1}, moviendo {swing_leg}")
        # ajustar stance para las otras patas
        for leg in stance_pos:
            if leg!=swing_leg: stance_pos[leg][0] -= advance_dy
        # puntos Bézier
        p0=stance_pos[swing_leg]
        p2=[p0[0]+2*advance_dy,p0[1]]
        p1=[(p0[0]+p2[0])/2,p0[1]+swing_h]
        # visualizar curva
        idx_hip = next(i for i,n in env.link_name_map.items() if n==leg_links[swing_leg][0])
        hip_w = np.array(p.getLinkState(env.robot, idx_hip, computeForwardKinematics=True)[4])
        pts = [ [hip_w[0], hip_w[1]+pt[0], hip_w[2]+pt[1]] for pt in [bezier(p0,p1,p2,t) for t in np.linspace(0,1,20)] ]
        for a,b in zip(pts,pts[1:]): p.addUserDebugLine(a,b,[0,1,0],2,step_time)
        # simulación
        for i in range(sim_steps):
            t=i/sim_steps
            action=np.zeros(12)
            # calcular y aplicar
            for leg, idxs in joint_idxs.items():
                foot_y,foot_z = (bezier(p0,p1,p2,t) if leg==swing_leg else stance_pos[leg])
                res = ik_leg_yz(foot_y, foot_z)
                if res is None: continue
                hip_a,knee_a = res
                # sumar offsets
                action[idxs[1]] = hip_a + initial_offsets[idxs[1]]
                action[idxs[2]] = knee_a + initial_offsets[idxs[2]]
            # paso
            obs,_,_,_,_ = env.step(action)
            demo_obs.append(obs); demo_act.append(action)
            # medir posiciones world
            for leg,(hip_name,toe_name) in leg_links.items():
                # log solo swing_leg
                if leg==swing_leg:
                    idx_toe = next(i for i,n in env.link_name_map.items() if n==toe_name)
                    toe_w = p.getLinkState(env.robot, idx_toe, computeForwardKinematics=True)[4]
                    hip_idx, knee_idx = joint_idxs[leg][1], joint_idxs[leg][2]
                    hip_ang=action[hip_idx]; knee_ang=action[knee_idx]
                    with open(log_path,'a') as f:
                        f.write(f"{step_count},{leg},{toe_w[1]:.4f},{toe_w[2]:.4f},{hip_ang:.4f},{knee_ang:.4f}\n")
            step_count+=1
            time.sleep(1/240.0)

# 7) Guardar dataset
np.savez(os.path.join(demo_dir,'expert_crawl_demos.npz'),
         observations=np.array(demo_obs), actions=np.array(demo_act))
print(f"✅ Datos y log guardados. Archivo de log: {log_path}")
env.close()
