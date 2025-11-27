# Genesis Joint Names (29 DOF order)
GENESIS_JOINT_NAMES_29 = [
    "left_hip_pitch_joint",      "right_hip_pitch_joint",     "waist_yaw_joint",
    "left_hip_roll_joint",       "right_hip_roll_joint",      "waist_roll_joint",
    "left_hip_yaw_joint",        "right_hip_yaw_joint",       "waist_pitch_joint",
    "left_knee_joint",           "right_knee_joint",          "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint","left_ankle_pitch_joint",    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",  "right_shoulder_roll_joint", "left_ankle_roll_joint",
    "right_ankle_roll_joint",    "left_shoulder_yaw_joint",   "right_shoulder_yaw_joint",
    "left_elbow_joint",          "right_elbow_joint",         "left_wrist_roll_joint",
    "right_wrist_roll_joint",    "left_wrist_pitch_joint",    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",      "right_wrist_yaw_joint",
]

# Low-level Policy Input Order (27 DOF - MuJoCo like)
LOW_LEVEL_JOINT_NAMES_27 = [
    'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint',
    'left_ankle_pitch_joint', 'left_ankle_roll_joint', 'right_hip_pitch_joint', 'right_hip_roll_joint',
    'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
    'waist_yaw_joint', 'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint',
    'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
    'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint',
    'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint'
]

# Low-level Policy Output Order (12 Leg Joints)
LOW_LEVEL_OUTPUT_LEG_12 = [
    'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint',
    'left_ankle_pitch_joint', 'left_ankle_roll_joint', 'right_hip_pitch_joint', 'right_hip_roll_joint',
    'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint'
]

# Left/Right Arm Joints
LEFT_ARM_JOINT_NAMES = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
]
RIGHT_ARM_JOINT_NAMES = [
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

# Default Joint Positions (Sparse definitions)
ISAAC_DEFAULTS_DICT = {
    0: -0.2, 1: -0.2, 9: 0.42, 10: 0.42, 13: -0.23, 14: -0.23,
    15: 0.23, 16: -0.23, 19: -0.1, 20: 0.1, 23: -0.32, 24: 0.32,
}

LOW_LEVEL_DEFAULTS_DICT = {
    0: -0.2, 3: 0.42, 4: -0.23, 6: -0.2, 9: 0.42, 10: -0.23,
    14: 0.23, 15: -0.1, 17: -0.32, 21: -0.23, 22: 0.1, 24: 0.32,
}

GENESIS_DEFAULTS_DICT = {
    'left_hip_pitch_joint': -0.20, 'right_hip_pitch_joint': -0.20,
    'left_knee_joint': 0.42, 'right_knee_joint': 0.42,
    'left_ankle_pitch_joint': -0.23, 'right_ankle_pitch_joint': -0.23,
    'left_shoulder_roll_joint': 0.23, 'right_shoulder_roll_joint': -0.23,
    'left_shoulder_yaw_joint': -0.1, 'right_shoulder_yaw_joint': 0.1,
    'left_wrist_roll_joint': -0.32, 'right_wrist_roll_joint': 0.32,
}

# Observation and Action Scales
ANG_VEL_SCALE = 0.5
DOF_POS_SCALE = 1.0
DOF_VEL_SCALE = 0.05
CMD_SCALE = [2.0, 2.0, 1.0]
HEIGHT_CMD_SCALE = 1.0
POLICY_ACTION_SCALE_LEGS = 0.25
HIERARCHICAL_ACTION_SCALE = 1.0
WAIST_ACTION_SCALE = 0.25
ARM_IK_ACTION_SCALE = 0.01

# Observation History
OBS_HISTORY_LEN = 6
SINGLE_OBS_DIM = 76

# Box Configuration
BOX_SIZE = (0.25, 0.25, 0.3)
BOX_STANDING_OFFSET = [-0.3, 0.0, 0.0]

# Default Paths (can be overridden)
DEFAULT_ROBOT_XML = 'assets/g1.xml'
DEFAULT_POLICY_PATH = 'assets/low_level.pt'
DEFAULT_HIGH_POLICY_PATH = 'assets/high_level.jit'

def get_pd_gains(joint_names):
    """Returns KP and KD lists based on joint names."""
    kp_values, kd_values = [], []
    for name in joint_names:
        if "hip_yaw" in name:      kp, kd = 100.0, 2.0
        elif "hip_roll" in name:   kp, kd = 100.0, 2.0
        elif "hip_pitch" in name:  kp, kd = 100.0, 2.0
        elif "knee" in name:       kp, kd = 150.0, 4.0
        elif "ankle" in name:      kp, kd = 40.0, 2.0
        elif "waist" in name:      kp, kd = 300.0, 5.0
        elif "shoulder" in name:   kp, kd = 200.0, 4.0
        elif "elbow" in name:      kp, kd = 100.0, 1.0
        elif "wrist" in name:      kp, kd = 20.0, 0.5
        else:                      kp, kd = 100.0, 2.0
        kp_values.append(kp)
        kd_values.append(kd)
    return kp_values, kd_values