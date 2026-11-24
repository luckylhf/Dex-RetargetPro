"""Target-specific geometry, source mapping, and executable speed limits."""

from math import isfinite
from typing import NamedTuple

ROBOT_ID = 'tienkung_dex_29'
ASSET_DIRECTORY = 'assets/dex_evt'
MODEL_XML = 'mjcf/dex_29.xml'
MODEL_URDF = 'urdf/dex_29.urdf'
MODEL_MANIFEST = 'model.json'
VELOCITY_SOURCE = 'URDF joint limit velocity, cross-checked against model.json'
LINEAR_SPEED_BASIS = (
    'Single axis at rated speed; neutral MJCF Jacobian, plus a geometric upper bound '
    'using first perpendicular lever arm and downstream segment lengths.'
)

# The floating base has no actuator speed rating in this asset. Supply a value
# here only for a robot with an independently established base envelope.
ROOT_LINEAR_LIMIT_M_S = None
ROOT_ANGULAR_LIMIT_RAD_S = None

# Reverse each leg with the foot frame held fixed. Neutral peaks enumerate
# the 64 corners of the six motor-speed intervals at zero hinge angles.
# The geometric bound sums the reverse-chain lever arms times rated speed
# (linear), or sums the six speed ratings (angular), rounding upwards.
# Each row is (neutral m/s, neutral rad/s, upper m/s, upper rad/s).
PELVIS_RELATIVE_TO_FOOT = {
    'Left': (28.472322, 54.556198, 43.576842, 86.707958),
    'Right': (28.472322, 54.556198, 43.576842, 86.707958),
}

# Independent project tolerances, retaining the 1.0.3 effective settings.
# The velocity allowance is converted with max(fps, 30 Hz).
ROOT_NEIGHBOR_JUMP_FLOOR_M = .035
ROOT_EXTRA_DROP_TOLERANCE_M = .035
ROOT_SPEED_ALLOWANCE_STEP_M = .035

PELVIS_BODY = 'pelvis'
TORSO_BODY = 'waist_pitch_link'
HEAD_BODY = 'head_yaw_link'
TORSO_SUPPORT_BODIES = (TORSO_BODY, 'camera_body_front_link', 'imu_waist_link')
WAIST_JOINTS = ('waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint')
WAIST_MEASUREMENT = 'shoulder_pitch_midpoint'
WAIST_MEASUREMENT_JOINTS = ('shoulder_pitch_l_joint', 'shoulder_pitch_r_joint')
SOLE_BAND_M = .006
SOLE_LENGTH_PERCENTILES = (5, 95)

LEG_AXES = ('hip_pitch', 'hip_roll', 'hip_yaw', 'knee_pitch', 'ankle_pitch', 'ankle_roll')
ARM_AXES = ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow_pitch',
            'elbow_yaw', 'wrist_pitch', 'wrist_roll')
SIDE_BODY_STEMS = dict(hip='hip_pitch', thigh='hip_yaw', knee='knee_pitch', foot='ankle_roll',
                       shoulder='shoulder_yaw', elbow='elbow_pitch', forearm='elbow_yaw',
                       wrist='wrist_pitch', hand='wrist_roll')
SIDE_LAYOUT = {}
for side, suffix in (('Left', 'l'), ('Right', 'r')):
    SIDE_LAYOUT[side] = dict(
        leg=tuple(f'{axis}_{suffix}_joint' for axis in LEG_AXES),
        arm=tuple(f'{axis}_{suffix}_joint' for axis in ARM_AXES),
        bodies={role: f'{stem}_{suffix}_link' for role, stem in SIDE_BODY_STEMS.items()}
        | {'tcp': f'{side.lower()}_tcp_link'},
        sole_geom=f'ankle_roll_{suffix}_link_visual',
        ankle_mount_visual=f'ankle_pitch_{suffix}_link_visual',
        ankle_mount_collision=f'ankle_pitch_{suffix}_link_collision',
        shin_collision=f'knee_pitch_{suffix}_link_collision',
    )


class JointMotionLimit(NamedTuple):
    angular_rad_s: float
    linear_neutral_m_s: float
    linear_geometric_upper_m_s: float
    point: str


# Measurement point meanings:
# *_sole: midpoint of the measured heel and toe on the robot foot visual mesh.
# *_tcp: origin of the configured hand TCP body.
# shoulder_pitch_midpoint: midpoint of the shoulder-pitch joint anchors,
# rigidly fixed in the torso. All three waist axes are relative to the pelvis.
# Linear speeds describe the contribution from this axis alone. They never
# cap global point speed, which also includes base and other joints' motion.
# At neutral, v = omega * ||axis_unit x (point - joint_anchor)||.
# Over other poses, replace that arm with the first perpendicular lever arm
# plus the remaining chain lengths. This follows the triangle inequality.
# Rounded to 6 decimals in m/s. A zero neutral value means alignment with
# the joint axis at neutral, even when the geometric upper bound is positive.
JOINT_SPEED_LIMITS = {
    'hip_pitch_l_joint': JointMotionLimit(16.755160819145562, 14.334951, 14.685844, 'Left_sole'),
    'hip_roll_l_joint': JointMotionLimit(16.755160819145562, 14.095539, 14.234025, 'Left_sole'),
    'hip_yaw_l_joint': JointMotionLimit(13.82300767579509, 0.252186, 6.603179, 'Left_sole'),
    'knee_pitch_l_joint': JointMotionLimit(11.100294042683936, 5.082127, 5.159060, 'Left_sole'),
    'ankle_pitch_l_joint': JointMotionLimit(14.137166941154069, 0.915604, 0.915634, 'Left_sole'),
    'ankle_roll_l_joint': JointMotionLimit(14.137166941154069, 0.802677, 0.802677, 'Left_sole'),
    'hip_pitch_r_joint': JointMotionLimit(16.755160819145562, 14.338519, 14.680712, 'Right_sole'),
    'hip_roll_r_joint': JointMotionLimit(16.755160819145562, 14.095946, 14.228892, 'Right_sole'),
    'hip_yaw_r_joint': JointMotionLimit(13.82300767579509, 0.242720, 6.598945, 'Right_sole'),
    'knee_pitch_r_joint': JointMotionLimit(11.100294042683936, 5.081881, 5.155659, 'Right_sole'),
    'ankle_pitch_r_joint': JointMotionLimit(14.137166941154069, 0.911257, 0.911303, 'Right_sole'),
    'ankle_roll_r_joint': JointMotionLimit(14.137166941154069, 0.803037, 0.803037, 'Right_sole'),
    'waist_yaw_joint': JointMotionLimit(10.053096491487338, 0.000000, 2.664714, 'shoulder_pitch_midpoint'),
    'waist_roll_joint': JointMotionLimit(13.82300767579509, 3.663982, 3.663982, 'shoulder_pitch_midpoint'),
    'waist_pitch_joint': JointMotionLimit(13.82300767579509, 2.579926, 2.579926, 'shoulder_pitch_midpoint'),
    'shoulder_pitch_l_joint': JointMotionLimit(7.218332720398149, 4.074027, 4.096793, 'Left_tcp'),
    'shoulder_roll_l_joint': JointMotionLimit(7.218332720398149, 4.074027, 4.096793, 'Left_tcp'),
    'shoulder_yaw_l_joint': JointMotionLimit(11.455294012539582, 0.000000, 3.737927, 'Left_tcp'),
    'elbow_pitch_l_joint': JointMotionLimit(11.455294012539582, 3.494510, 3.508821, 'Left_tcp'),
    'elbow_yaw_l_joint': JointMotionLimit(18.325957145940457, 0.000000, 1.942551, 'Left_tcp'),
    'wrist_pitch_l_joint': JointMotionLimit(18.325957145940457, 1.942551, 1.942551, 'Left_tcp'),
    'wrist_roll_l_joint': JointMotionLimit(18.325957145940457, 0.806342, 0.806342, 'Left_tcp'),
    'shoulder_pitch_r_joint': JointMotionLimit(7.218332720398149, 4.074027, 4.096793, 'Right_tcp'),
    'shoulder_roll_r_joint': JointMotionLimit(7.218332720398149, 4.074027, 4.096793, 'Right_tcp'),
    'shoulder_yaw_r_joint': JointMotionLimit(11.455294012539582, 0.000000, 3.737927, 'Right_tcp'),
    'elbow_pitch_r_joint': JointMotionLimit(11.455294012539582, 3.494510, 3.508821, 'Right_tcp'),
    'elbow_yaw_r_joint': JointMotionLimit(18.325957145940457, 0.000000, 1.942551, 'Right_tcp'),
    'wrist_pitch_r_joint': JointMotionLimit(18.325957145940457, 1.942551, 1.942551, 'Right_tcp'),
    'wrist_roll_r_joint': JointMotionLimit(18.325957145940457, 0.806342, 0.806342, 'Right_tcp'),
}
JOINT_ORDER = (*SIDE_LAYOUT['Left']['leg'], *SIDE_LAYOUT['Right']['leg'], *WAIST_JOINTS,
               *SIDE_LAYOUT['Left']['arm'], *SIDE_LAYOUT['Right']['arm'])


def joint_angular_limits(names, manifest=None):
    limits = [JOINT_SPEED_LIMITS[name].angular_rad_s for name in names]
    for name, limit in zip(names, limits):
        if not isfinite(limit) or limit <= 0:
            raise ValueError(f'Invalid angular speed limit for {name}: {limit}')
        if manifest is not None:
            rated = float(manifest['joint_limits'][name]['velocity'])
            if not isfinite(rated) or not 0 < limit <= rated + 1e-10:
                raise ValueError(f'Configured speed for {name} exceeds the supplied asset rating')
    return limits


# The canonical source landmarks consumed by contact preparation and IK.
SMPLX_BODY_JOINTS = dict(Hips='pelvis', Spine='spine1', Neck='neck', Head='head')
SMPLX_LIMB_JOINTS = dict(UpLeg='hip', Leg='knee', Foot='ankle', ToeBase='foot',
                         Arm='shoulder', ForeArm='elbow', Hand='wrist')
SMPLX_FINGER_JOINTS = dict(distal='middle1', thumb_side='index1', little_side='pinky1')
SMPLX_Y_UP_TO_Z_UP = ((1., 0., 0.), (0., 0., -1.), (0., 1., 0.))
SMPLX_PELVIS_LOCAL_UP = (0., 1., 0.)
SMPLX_ANKLE_LOCAL_UP = (0., 1., 0.)
SMPLX_ELBOW_CALIBRATION_RANGE_RAD = (.35, 2.5)
SMPLX_ELBOW_FALLBACK = {'Left': (0., 1., 0.), 'Right': (0., -1., 0.)}
SMPLX_SOLE_SUFFIXES = dict(all='_all', heel='_heel', toe='_toe', corners='_sole_corners')
SMPLX_FOOT_SKIN_WEIGHT = .5
SMPLX_SOLE_BAND_M = .012
SMPLX_SOLE_PERCENTILES = (25, 75)

# Target link-local directions. The torso and foot semantic bases coincide
# with their link-local XYZ frames in the shipped MJCF.
TCP_FORWARD = (0., 0., -1.)
TCP_WIDTH = (1., 0., 0.)
FOREARM_WIDTH = (1., 0., 0.)
ELBOW_FLEXION_AXIS = (0., -1., 0.)
ELBOW_BEND_SIGN = -1.

# Collision names describe this asset's assembly and mesh conventions.
ASSEMBLY_COLLISION_EXCLUSIONS = (('pelvis_collision', 'waist_roll_link_collision'),)
COLLISION_SUFFIX = '_collision'
COLLISION_TORSO_EXCLUDED_STEMS = ('shoulder', 'ankle', 'knee')

# NASA medical screening references inform the wrist preference and envelope
# before intersecting with the robot limits. These are ranges of motion in
# degrees, separate from the angular speed limits above.
NASA_HAND_REFERENCE = 'NASA OCHMO-STD-100.1A Rev A, page 82'
NASA_HAND_REFERENCE_URL = 'https://www.nasa.gov/wp-content/uploads/2023/04/ochmo-std-100.1a.pdf#page=82'
NASA_WRIST_ROM_DEG = (
    ('radial', 15, 20), ('ulnar', 30, 45),
    ('flexion', 70, 80), ('extension', 65, 85),
)
NASA_FOREARM_ROM_DEG = (('supination_neutral', 80, 90), ('pronation_neutral', 80, 90))
NASA_HAND_FUNCTIONS = (
    'composite_grip', 'full_finger_extension', 'intrinsic_and_thenar_muscle_condition',
    'thumb_to_finger_opposition',
)
WRIST_AXES = {
    'Left': (('radial', 'ulnar'), ('flexion', 'extension')),
    'Right': (('radial', 'ulnar'), ('extension', 'flexion')),
}
WRIST_RELATIVE_FRAME = 'forearm-to-hand Ry(J6) Rx(J7), neutral mounting verified'


def wrist_profile(overrides=None):
    """Return independent wrist defaults for a conversion or replay."""
    return dict(
        reference=NASA_HAND_REFERENCE,
        url=NASA_HAND_REFERENCE_URL,
        soft_deg={axis: lower for axis, lower, _ in NASA_WRIST_ROM_DEG},
        maximum_deg={axis: upper for axis, _, upper in NASA_WRIST_ROM_DEG},
    ) | (overrides or {})
