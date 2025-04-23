import pybullet as p
import pybullet_data
import numpy as np

def test_robot_initial_joint_angles():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)

    p.loadURDF("plane.urdf")

    #
    #baseOrientation=[0, 0.5, 0.5, 0]

    #"D:\\ITMO trabajos de la u\\tesis\\py\\testing\\RL\\unitree_ros-master\\robots\\laikago_description\\urdf\\laikago.urdf"
    robot= p.loadURDF(
        "D:\\ITMO trabajos de la u\\tesis\py\\testing\\pybullet_robots\\data\\laikago\\laikago_toes.urdf",
        basePosition=[0, 0, 0.5],baseOrientation=[0, 0.5, 0.5, 0],    
        useFixedBase=False
    )

    # Ángulos iniciales (grados → radianes)
    q_deg = np.tile([0, -45, -45], 4)
    q_rad = np.radians(q_deg)
    
        # ids de articulaciones revolutas
    jids = [i for i in range(p.getNumJoints(robot))
            if p.getJointInfo(robot, i)[2] == p.JOINT_REVOLUTE]

    # estado inicial
    for jid, q in zip(jids, q_rad):
        p.resetJointState(robot, jid, q)

    # mantenlos con control de posición
    p.setJointMotorControlArray(robot, jids,
                                controlMode=p.POSITION_CONTROL,
                                targetPositions=q_rad,
                                positionGains=[0.4]*len(jids),
                                forces=[60]*len(jids))


    for _ in range(240*10):
        p.stepSimulation()

    # ------------------  Resultados  ------------------
    base_pos, _ = p.getBasePositionAndOrientation(robot)
    altura_instantanea = base_pos[2]
    
    print(f"Altura instantánea al terminar: {altura_instantanea:.4f} m")
    
    p.disconnect()

if __name__ == "__main__":
    test_robot_initial_joint_angles()
