"""Configuration of a1 robot"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
### from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
import os

##
# Configuration
##


A1_CFG = ArticulationCfg(
    prim_path = "/World/envs/env_.*/robot",
    spawn = sim_utils.UsdFileCfg(
        usd_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "a1_ur12e.usd"),
        activate_contact_sensors = True,
        rigid_props = sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled = True,
            disable_gravity = True,
            max_depenetration_velocity = 5.0,
        ),
        articulation_props = sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions = True,
            solver_position_iteration_count = 12,
            solver_velocity_iteration_count = 1,
        ),
        collision_props = sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        semantic_tags = [("class", "none")]
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos = (0.0, 0.0, 0.0),
        rot = (1, 0, 0, 0),
        joint_pos={
            # "xiaoyu_arm_joint1": 1.5707824230194092,
            # "xiaoyu_arm_joint2": -2.3387411976724017,
            # "xiaoyu_arm_joint3": 1.6580627893946132,
            # "xiaoyu_arm_joint4": -1.9040199718871058,
            # "xiaoyu_arm_joint5": -1.5707963267948966,
            # "xiaoyu_arm_joint6": 0.0,
            "xiaoyu_arm_joint1": 1.5707824230194092,
            "xiaoyu_arm_joint2": -2.0071660480894984,
            "xiaoyu_arm_joint3": 1.3613484541522425,
            "xiaoyu_arm_joint4": -0.9599629205516357,
            "xiaoyu_arm_joint5": -1.570770565663473,
            "xiaoyu_arm_joint6": 0.0,
        },
    ),
    actuators={
        "a1_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["xiaoyu_arm_joint[1-3]"],
            effort_limit=150,
            velocity_limit=3.141592653589793,
            stiffness=0,
            damping=10,
        ),
        "a1_arm": ImplicitActuatorCfg(
            joint_names_expr=["xiaoyu_arm_joint[4-6]"],
            effort_limit=28,
            velocity_limit=3.141592653589793,
            stiffness=0,
            damping=10,
        ),
    },
)


"""Configuration of a1 robot"""