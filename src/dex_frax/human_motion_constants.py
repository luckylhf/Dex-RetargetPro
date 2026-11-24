"""Human reference measurements, with provenance."""

from math import radians

# Keep the legacy import working while the target profile owns wrist settings.
from .robot_profile import wrist_profile

# Graubner & Nixdorf (2011), Table D, Vmax column: 12.34 m/s at 67.90 m
# during Bolt's 2009 Berlin 100 m final. The detector uses this observed
# horizontal sprint peak as a review reference for declared unaided ground running.
HUMAN_SPRINT_PEAK_M_S = 12.34
HUMAN_SPRINT_REFERENCE = (
    'https://worldathletics.org/download/downloadnsa?filename=31514203-6724-455e-906c-ef931384c656.pdf'
    '&urlslug=biomechanical-analysis-of-the-sprint-and-hurd'
)

# The Relationship of Range of Motion, Hip Shoulder Separation, and Pitching
# Kinematics (2020), Table 2: mean ± SD of peak pelvis rotation rates during
# high-school baseball pitching, 669.1 ± 95.5 deg/s. Activity-specific data.
HUMAN_PITCHING_PELVIS_PEAK_MEAN_RAD_S = radians(669.1)
HUMAN_PITCHING_PELVIS_PEAK_SD_RAD_S = radians(95.5)
HUMAN_PITCHING_REFERENCE = 'https://pmc.ncbi.nlm.nih.gov/articles/PMC7727427/'

# Willer et al. (2024), Table 3: mean ± SD of peak knee extension rates during
# the swing phase of treadmill running at 8.33 m/s, 1197 ± 65 deg/s. The
# study reports a group mean for that specific activity.
HUMAN_RUNNING_SWING_KNEE_EXTENSION_PEAK_MEAN_RAD_S = radians(1197.)
HUMAN_RUNNING_SWING_KNEE_EXTENSION_PEAK_SD_RAD_S = radians(65.)
HUMAN_RUNNING_REFERENCE = 'https://onlinelibrary.wiley.com/doi/full/10.1111/sms.14690'


# Guinness, Olivia Oliver, 2015: whole-body ice spin, 342 rpm x 6 deg/s.
# This approximates the pelvis rate only during coordinated upright yaw.
HUMAN_PELVIS_SPIN_REVIEW_RAD_S = radians(342. * 360. / 60.)
HUMAN_PELVIS_SPIN_REFERENCE = 'https://www.guinnessworldrecords.com/world-records/fastest-spin-ice-skating'

# Bastiaansen et al. (2024), Fig. 6: highest individual point about 2170–2180
# deg/s in full-effort instep kicks. 2200 is a rounded review reference.
HUMAN_KNEE_EXTENSION_REVIEW_RAD_S = radians(2200.)
HUMAN_KNEE_KICK_REFERENCE = 'https://link.springer.com/article/10.1007/s12283-024-00458-4#Fig6'

# Engineering review line for world speed of hand/foot landmarks. The
# measurements below motivate the scale; they do not establish 25 m/s.
HUMAN_EXTREMITY_REVIEW_M_S = 25.
HUMAN_HAND_SPEED_REFERENCE = 'https://ojs.ub.uni-konstanz.de/cpa/article/download/4491/4179'
HUMAN_FOOT_SPEED_REFERENCE = 'https://www.jssm.org/volume01/iss3/cap/jssm-01-72.pdf'

# An entry is keyed by the speed check and must specify threshold (in the
# check's units), source, axis and scope (the check's applicability).
# The measurements above support review only, so there are no hard entries.
HUMAN_HARD_SPEED_LIMITS = {}


def source_reference_scope():
    """Provenance and usable domains of the references shipped with this release."""
    return {
        'fixed_lafan_channels': {'kind':'source_rig_constraint',
            'scope':'non-root local translations and rigid LaFAN toes'},
        'horizontal_sprint': {'kind':'activity_measurement',
            'metres_per_second':HUMAN_SPRINT_PEAK_M_S, 'source':HUMAN_SPRINT_REFERENCE,
            'measurement':'Graubner and Nixdorf (2011), Table D: Bolt, Berlin 2009 100 m final, Vmax',
            'scope':'Root horizontal speed during declared unaided ground running; review only'},
        'pelvis_spin_review': {'kind':'activity_proxy', 'rad_s':HUMAN_PELVIS_SPIN_REVIEW_RAD_S,
            'source':HUMAN_PELVIS_SPIN_REFERENCE, 'scope':'Coherent upright pelvic and torso yaw; review only'},
        'knee_extension_review': {'kind':'rounded_activity_reference',
            'rad_s':HUMAN_KNEE_EXTENSION_REVIEW_RAD_S, 'source':HUMAN_KNEE_KICK_REFERENCE,
            'scope':'Extension around estimated anatomical flexion axis; review only'},
        'extremity_world_speed_review': {'kind':'engineering_review', 'metres_per_second':HUMAN_EXTREMITY_REVIEW_M_S,
            'sources':[HUMAN_HAND_SPEED_REFERENCE, HUMAN_FOOT_SPEED_REFERENCE],
            'scope':'Hand and foot world landmark speed; review only'},
        'pitching_pelvis': {'kind':'activity_group_mean',
            'mean_rad_s':HUMAN_PITCHING_PELVIS_PEAK_MEAN_RAD_S,
            'sd_rad_s':HUMAN_PITCHING_PELVIS_PEAK_SD_RAD_S, 'source':HUMAN_PITCHING_REFERENCE,
            'measurement':'Table 2: peak pelvis rotation in high-school baseball pitching',
            'scope':'Activity-specific group mean and standard deviation'},
        'running_knee_extension': {'kind':'activity_group_mean',
            'mean_rad_s':HUMAN_RUNNING_SWING_KNEE_EXTENSION_PEAK_MEAN_RAD_S,
            'sd_rad_s':HUMAN_RUNNING_SWING_KNEE_EXTENSION_PEAK_SD_RAD_S, 'source':HUMAN_RUNNING_REFERENCE,
            'measurement':'Willer et al. (2024), Table 3: peak swing knee extension at 8.33 m/s',
            'scope':'Activity-specific group mean and standard deviation'},
        'wrist_rom': {'kind':'medical_screening_angle_reference', 'profile':wrist_profile(),
            'scope':'NASA Rev A, p. 82, item 11C: medical screening ranges in degrees',
            'use':'target wrist envelope; no validated native anatomical wrist-axis calibration'},
        'population_wide_speed_limits_by_joint':None,
    }
