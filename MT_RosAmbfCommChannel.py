import random
# Import the relevant classes
import rospy
import tf2_ros
from kils import Pose
import PyKDL
import reflexxes
import numpy as np
import threading
import subprocess
import sys

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from PyKDL import Frame, Rotation, Vector
import time
from enum import Enum
import numpy as np
from std_msgs.msg import Empty
from sensor_msgs.msg import Image
from surgical_robotics_challenge.task_completion_report import TaskCompletionReport
import surgical_robotics_challenge.kinematics.psmIK as psm_ik
import geometry_msgs.msg
import sensor_msgs.msg
#from surgical_robotics_challenge.simulation_manager import SimulationManager, SimulationObject
from surgical_robotics_challenge.simulation_manager import SimulationManager
from ambf_msgs.msg import RigidBodyCmd, RigidBodyState
from ambf_client import Client

import collections
from gym import spaces
import time
from datetime import datetime
import subprocess
import sys
import csv
# Connect to AMBF and setup image subscriber
import img_saver as ImgSaver

import GetEstimation


# TODO: Clean up imports


RATE_HZ = 120
NEEDLE_RADIUS = 0.1018 / 10
THETA_GRASP = np.radians(15)


def add_break(s):
    time.sleep(s)
    print('-------------')

class ImageSub:
    def __init__(self, image_topic):
        self.image_sub = rospy.Subscriber(image_topic, Image, self.image_cb)
        self.image_msg = Image()

    def image_cb(self, image_msg):
        self.image_msg = image_msg


class ArmType(Enum):
    PSM1=1
    PSM2=2
    ECM=3

def frame_to_pose_stamped_msg(frame):
    msg = PoseStamped()
    msg.header.stamp = rospy.Time.now()
    msg.pose.position.x = frame.p[0]
    msg.pose.position.y = frame.p[1]
    msg.pose.position.z = frame.p[2]

    msg.pose.orientation.x = frame.M.GetQuaternion()[0]
    msg.pose.orientation.y = frame.M.GetQuaternion()[1]
    msg.pose.orientation.z = frame.M.GetQuaternion()[2]
    msg.pose.orientation.w = frame.M.GetQuaternion()[3]

    return msg


def list_to_sensor_msg_position(jp_list):
    msg = JointState()
    msg.position = jp_list
    return msg

def pose_to_kdl_frame(pose):
    return PyKDL.Frame(
        PyKDL.Rotation.Quaternion(pose.q.x, pose.q.y, pose.q.z, pose.q.w),
        PyKDL.Vector(pose.p[0], pose.p[1], pose.p[2])
    )

def pos_to_action(pose):
    return [pose.p[0], pose.p[1], pose.p[2], pose.q.x, pose.q.y, pose.q.z, pose.q.w]

class TFBuffer:
    def __init__(self):
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

    def lookup_pose(self, target_frame, source_frame):
        msg = self.tf_buffer.lookup_transform(target_frame, source_frame, rospy.Time(), rospy.Duration(3))
        return Pose.from_msg(msg.transform)

def publish_helper_frames(tf_static_broadcaster):
    transforms = []

    # Needle base (end) frame
    t_needle_base = Pose(position=[-NEEDLE_RADIUS, 0, 0])
    m = geometry_msgs.msg.TransformStamped()
    m.header.stamp = rospy.Time.now()
    m.header.frame_id = "Needle"
    m.child_frame_id = "Needle/Base"
    m.transform = t_needle_base.to_msg(geometry_msgs.msg.Transform)
    transforms.append(m)

    # Needle tip frame
    t_needle_tip = Pose.from_axis_angle([0, 0, 1], -np.radians(90 + 33.5)) @ Pose(position=[-NEEDLE_RADIUS, 0, 0])
    m = geometry_msgs.msg.TransformStamped()
    m.header.stamp = rospy.Time.now()
    m.header.frame_id = "Needle"
    m.child_frame_id = "Needle/Tip"
    m.transform = t_needle_tip.to_msg(geometry_msgs.msg.Transform)
    transforms.append(m)

    # Needle grasp frame
    flip = Pose.from_matrix(
        np.array([
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
        ])
    )

    # This is currently just a static frame and so will not work
    # with arbitrary needle orientations. Preferably we'd re-compute
    # grasp pose every time one is needed.
    t_needle_grasp = Pose.from_axis_angle([0, 0, 1], -THETA_GRASP) @ Pose(position=[-NEEDLE_RADIUS, 0, 0]) @ flip
    m = geometry_msgs.msg.TransformStamped()
    m.header.stamp = rospy.Time.now()
    m.header.frame_id = "Needle"
    m.child_frame_id = "Needle/Grasp"
    m.transform = t_needle_grasp.to_msg(geometry_msgs.msg.Transform)
    transforms.append(m)

    tf_static_broadcaster.sendTransform(transforms)



class ARMInterface:
    def __init__(self, arm_type):
        if arm_type == ArmType.PSM1:
            arm_name = '/CRTK/psm1'
        elif arm_type == ArmType.PSM2:
            arm_name = '/CRTK/psm2'
        elif arm_type == ArmType.ECM:
            arm_name = '/CRTK/ecm'
        else:
            raise ("Error! Invalid Arm Type")

        self._cp_sub = rospy.Subscriber(arm_name + "/measured_cp", PoseStamped, self.cp_cb, queue_size=1)
        self._T_b_w_sub = rospy.Subscriber(arm_name + "/T_b_w", PoseStamped, self.T_b_w_cb, queue_size=1)
        self._jp_sub = rospy.Subscriber(arm_name + "/measured_cp", JointState, self.jp_cb, queue_size=1)
        self.subscriber = rospy.Subscriber(arm_name + "/measured_cp", PoseStamped, self.set_measured_cp, queue_size=1)
        self.cp_pub = rospy.Publisher(arm_name + "/servo_cp", PoseStamped, queue_size=1)
        self.jp_pub = rospy.Publisher(arm_name + "/servo_jp", JointState, queue_size=1)
        self.jaw_jp_pub = rospy.Publisher(arm_name + '/jaw/' + 'servo_jp', JointState, queue_size=1)

        self.subscribers = [
            rospy.Subscriber(arm_name + '/measured_cp', geometry_msgs.msg.PoseStamped, self.set_measured_cp, queue_size=1),
            rospy.Subscriber(arm_name + '/measured_js', sensor_msgs.msg.JointState, self.set_measured_js, queue_size=1),
        ]

        self.measured_cp_msg = None
        self.T_b_w_msg = None
        self.measured_jp_msg = None

        self._jp_sub = rospy.Subscriber(arm_name + "/measured_cp", JointState, self.jp_cb, queue_size=1)

        self.trajectory = None
        self.traj_done_condition = threading.Condition(threading.Lock())
        max_vel = [1, 1, 1, 1, 1, 1] #[np.pi / 4, np.pi / 4, 0.05, np.pi / 4, np.pi / 4, np.pi / 4]
        self.traj_gen = reflexxes.extra.PositionTrajectoryGenerator(
            number_of_dofs=6,
            cycle_time=1 / RATE_HZ,
            max_velocity=max_vel,
            max_acceleration=[4 * x for x in max_vel],
            max_jerk=[16 * x for x in max_vel],
        )
        # self.traj_gen.flags.SynchronizationBehavior = reflexxes.RMLPositionFlags.NO_SYNCHRONIZATION
        self.timer = rospy.Timer(rospy.Duration(1 / RATE_HZ), self.step_control_loop)
        self.measured_jp_msg = None

    def set_measured_cp(self, msg):
        self.msg_measured_cp = msg

    def set_measured_js(self, msg):
        self.msg_measured_js = msg

    def cp_cb(self, msg):
        self.measured_cp_msg = msg

    def T_b_w_cb(self, msg):
        self.T_b_w_msg = msg

    def jp_cb(self, msg):
        self.measured_jp_msg = msg

    def measured_cp(self):
        return self.measured_cp_msg

    def get_T_b_w(self):
        return self.T_b_w_msg

    def measured_jp(self):
        return self.measured_jp_msg

    def servo_cp(self, pose):
        if type(pose) == Frame:
            msg = frame_to_pose_stamped_msg(pose)
        else:
            msg = pose
        self.cp_pub.publish(msg)

    def servo_jp(self, position):
        # if type(jp) == list:
        #     msg = list_to_sensor_msg_position(jp)
        # else:
        #     msg = jp
        # self.jp_pub.publish(msg)

        m = sensor_msgs.msg.JointState()
        m.header.stamp = rospy.Time.now()
        m.position = position
        #self.pub_servo_jp.publish(m)
        self.jp_pub.publish(m)


    def set_jaw_angle(self, val):
        msg = list_to_sensor_msg_position([val])
        self.jaw_jp_pub.publish(msg)

    def move_js_ik_sync(self, t_base_tcp_desired):
        if self.trajectory:
            rospy.logwarn("Already moving")
            return

        cv = self.move_js_ik(t_base_tcp_desired)

        with cv:
            cv.wait_for(lambda: self.trajectory is None)

        # rospy.sleep(0.2)

    def move_js_ik(self, t_base_tcp_desired):
        ik_solution = psm_ik.compute_IK(pose_to_kdl_frame(t_base_tcp_desired))
        # ik_solution = psm_ik.enforce_limits(ik_solution)
        self.traj_gen.current_position = self.msg_measured_js.position
        self.traj_gen.current_velocity = self.msg_measured_js.velocity

        with self.traj_done_condition:
            self.trajectory = self.traj_gen.trajectory(ik_solution, [0]*6)

        return self.traj_done_condition

    def step_control_loop(self, event):
        if not self.trajectory:
            return
        try:
            pos_joint_next = next(self.trajectory)[0]
            self.servo_jp(pos_joint_next)
        except StopIteration:
            with self.traj_done_condition:
                self.trajectory = None
                self.traj_done_condition.notify_all()


class SceneObjectType(Enum):
    Needle=1
    Entry1=2
    Entry2=3
    Entry3=4
    Entry4=5
    Exit1=6
    Exit2=7
    Exit3=8
    Exit4=9


class SceneInterface:
    def __init__(self):
        self._scene_object_poses = dict()
        self._scene_object_poses[SceneObjectType.Needle] = None
        self._scene_object_poses[SceneObjectType.Entry1] = None
        self._scene_object_poses[SceneObjectType.Entry2] = None
        self._scene_object_poses[SceneObjectType.Entry3] = None
        self._scene_object_poses[SceneObjectType.Entry4] = None
        self._scene_object_poses[SceneObjectType.Exit1] = None
        self._scene_object_poses[SceneObjectType.Exit2] = None
        self._scene_object_poses[SceneObjectType.Exit3] = None
        self._scene_object_poses[SceneObjectType.Exit4] = None

        self._subs = []
        self._pubs = []

        namespace = '/CRTK/'
        suffix = '/measured_cp'

        for k, i in self._scene_object_poses.items():
            self._subs.append(rospy.Subscriber(namespace + k.name + suffix, PoseStamped,
                                               self.state_cb, callback_args=k, queue_size=1))

        #topic = '/ambf/env/Needle/Command' #'/servo_cp'
        #/ambf/env/Needle/State
        self._Needle_pub = rospy.Publisher('/ambf/env/Needle/Command', RigidBodyCmd, queue_size=1)
        #self._Needle_pub = rospy.Publisher('/CRTK/Needle/measured_cp', PoseStamped, queue_size=1)

        self._task_3_ready = False
        self._task_3_setup_init_pub = rospy.Publisher('/CRTK/scene/task_3_setup/init', Empty, queue_size=1)

        self._task_3_setup_ready_sub = rospy.Subscriber('/CRTK/scene/task_3_setup/ready',
                                                        Empty, self.task_3_setup_ready_cb, queue_size=1)

    def state_cb(self, msg, key):
        self._scene_object_poses[key] = msg

    def measured_cp(self, object_type):
        return self._scene_object_poses[object_type]

    def task_3_setup_ready_cb(self, msg):
        self._task_3_ready = True

    def task_3_setup_init(self):
        self._task_3_ready = False
        self._task_3_setup_init_pub.publish(Empty())
        while not self._task_3_ready:
            time.sleep(0.1)

class WorldInterface:
    def __init__(self):
        self._reset_world_pub = rospy.Publisher('/ambf/env/World/Command/Reset', Empty, queue_size=1)
        self._reset_bodies_pub = rospy.Publisher('/ambf/env/World/Command/ResetBodies', Empty, queue_size=1)

    def reset(self):
        self._reset_world_pub.publish(Empty())

    def reset_bodies(self):
        self._reset_bodies_pub.publish(Empty())


class RosAmbfCommChannel(object):
    """ RosAmbfCommChannel:
        Wrapper that provides a communication channel between
        the surgical environment and tf-agents
     """

    def __init__(self):  # TODO: could use a clean up
        self.SM_STATE = "GOTO_NEEDLE"
        self.saver = ImgSaver.ImageSaver()
        self.vision_method = GetEstimation.estimation("AE")
        # Create an instance of the client
        rospy.init_node('RosAmbfCC')
        time.sleep(0.5)

        # Get world, for reset
        self.world_handle = WorldInterface()
        # Get a handle to PSM1
        self.psm1 = ARMInterface(ArmType.PSM1)
        # Get a handle  to PSM2
        self.psm2 = ARMInterface(ArmType.PSM2)
        # Get a handle to ECM
        self.ecm = ARMInterface(ArmType.ECM)
        # Get a handle to scene to access its elements, i.e. needle and entry / exit points
        self.scene = SceneInterface()
        # Small sleep to let the handles initialize properly
        add_break(0.5)

        # Add you camera stream subs
        self.cameraL_sub = ImageSub('/ambf/env/cameras/cameraL/ImageData')
        self.cameraR_sub = ImageSub('/ambf/env/cameras/cameraR/ImageData')

        # Set entry target
        self.entry_target_string = None
        self.pos_entry = None
        #self.set_target_entry(random.choice([1,2,3,4]))

        # Set up a transform buffer to manage coordinate frame transformations
        self.tf_buffer = TFBuffer()
        # Set up a static transform broadcaster to publish coordinate frame transformations
        self.tf_static_broadcaster = tf2_ros.StaticTransformBroadcaster()

        # Publish some helper frames to the coordinate frame tree
        publish_helper_frames(self.tf_static_broadcaster)
        self.t_base_grasp = self.tf_buffer.lookup_pose('psm2/baselink', 'Needle/Grasp')
        self.t_above_needle = pos_to_action(self.t_base_grasp @ Pose(position=[0, 0, -0.01]))  #random.choice([-0.015, -0.01, -0.005])
        #self.t_grasp_needle = pos_to_action(self.t_base_grasp @ Pose(position=[0, 0, 0.0045])) #0.004 #0.006  #random.choice([0.005, 0.006, 0.007])
        self.t_grasp_needle = pos_to_action(self.t_base_grasp @ Pose(position=[1, 0, 0.0045])) # TODO: try different offset in x [0.01],y
        self.t_tcp0_needletip = None
        self.t_base_tcp_desired = None

        self.current_action = self._get_psm2_ik()
        self._reached_frame_flag = False
        self.SM_STATE = "GOTO_NEEDLE"
        self.done = False
        self.prev_delta_norm = 1
        self.needle_entry_dist = 999
        self.needle_is_grasp = False

        self._average_needle_entry_dist = []
        self.AVG_needle_entry_dist = 999
        self._average_psm_needle_dist = []
        self.AVG_psm_needle_dist = 999

        self.Random_Needle_drop = False
        self.Has_dropped_once = False
        #self.accumulated_optimal_action_error_norm = 0
        self.grasp_steps = 0
        self.pickup_attempts = 0

        self.init_grasp = random.uniform(0.5, 1.0)
        self._psm1_angle = 1
        self._psm2_angle = 1

        self.pos_entry_target_number = 1



    def set_seed(self, seed=None):
        print(f'random seed set in wrapper: {seed}')
        random.seed(seed)

    def set_random_needle(self):
        # Look up the pose of the needle grasp location relative to the base link of the surgical arm
        t_base_grasp = self.tf_buffer.lookup_pose('psm2/baselink', 'Needle/Grasp')
        # Move the surgical arm to a position above the grasp location
        self.move_jaw('psm2', np.pi / 4)
        rospy.sleep(0.1)
        self.psm2.move_js_ik_sync(t_base_grasp @ Pose(position=[0, 0, -0.01]))  # above grasp
        rospy.sleep(0.1)
        # Move the surgical arm to a position at the grasp location, but slightly lower
        self.psm2.move_js_ik_sync(t_base_grasp @ Pose(position=[0.0, 0.0, 0.006]))
        rospy.sleep(0.1)
        self.move_jaw('psm2', 0.0)
        rospy.sleep(0.1)
        # Move the surgical arm to a position above the grasp location
        self.psm2.move_js_ik_sync(t_base_grasp @ Pose(position=[0, 0, -0.01]))
        rospy.sleep(0.1)

        # Move to random offset and drop needle
        domain_set = [[-0.035, 0.01],                                               # Domain 0: Fixed position
                      [random.uniform(-0.04, -0.03), random.uniform(0, 0.02)],      # Domain 1: Small displacement
                      [random.uniform(-0.06, -0.01), random.uniform(0, 0.02)],      # Domain 2: Medium displacement
                      [random.uniform(-0.08, 0.001), random.uniform(0, 0.015)],     # Domain 3: Entire right side
                      [random.uniform(-0.08, 0.001), random.uniform(0, 0.055)],     # Domain 4: Entire suture domain
                      [random.uniform(-0.06, 0.0), 0.01],                           # Domain 5: Right side Line (experimental)
                      ]

        rx, ry = domain_set[3]
        print(f'Random needle position: [{rx}, {ry}]')

        self.psm2.move_js_ik_sync(t_base_grasp @ Pose(position=[rx, ry, -0.01]))
        rospy.sleep(0.1)
        self.psm2.move_js_ik_sync(t_base_grasp @ Pose(position=[rx, ry, 0.005]))
        rospy.sleep(0.1)
        self.move_jaw('psm2', np.pi / 4)
        rospy.sleep(0.1)
        self.psm2.move_js_ik_sync(t_base_grasp @ Pose(position=[rx, ry, -0.01]))
        rospy.sleep(0.1)

    def set_random_psm(self):  # TODO: Implement random psm init
        return 0

    def Average(self, lst):
        return sum(lst) / len(lst)
    def _update_metrics(self):
        """ Calculates the needle entry distance """
        needle = self.scene.measured_cp(SceneObjectType.Needle).pose
        pos_needle = [needle.position.x, needle.position.y, needle.position.z, needle.orientation.x, needle.orientation.y, needle.orientation.z, needle.orientation.w]
        target_frame = []
        if self.entry_target_string == "Entry1":
            target_frame = [0.0032030713004311063, 0.0427724265194714, 0.08301865045707325, -0.7039961189585819, -0.05714372197092554, -0.06190763451622573, 0.7051889848254644]
        if self.entry_target_string == "Entry2":
            target_frame = [0.002319689448048434, 0.019586229727386675, 0.08267958824738855, -0.7047013476442332, -0.05274771595771238, -0.06254982835222055, 0.7047703229157382]
        if self.entry_target_string == "Entry3":
            target_frame = [0.003272679402915321, -0.003574436322023815, 0.08303560050551816, -0.7039081705099594, -0.049309026131210514, -0.04756115746676991, 0.7069793799904879]
        if self.entry_target_string == "Entry4":
            target_frame = [0.0018813501756928628, -0.02861604686303037, 0.08251535720090267, -0.7041061157709064, -0.05576485406222165, -0.06411373978484454, 0.7049924021975182]

        delta = np.array(target_frame, dtype=np.float32) - np.array(pos_needle, dtype=np.float32)
        self.needle_entry_dist = np.linalg.norm(delta[:3])

        """ Calculates the PSM2 to Needle distance """
        psm2_to_Needle = self.tf_buffer.lookup_pose('psm2/toolyawlink', 'Needle/Grasp')
        psm2_to_Needle = pos_to_action(psm2_to_Needle)

        _psm2_to_Needle = np.linalg.norm(psm2_to_Needle[:3])
        if _psm2_to_Needle <= 0.009 and self._psm2_angle == 0:
            self.needle_is_grasp = True
            self.grasp_steps += 1
        else:
            self.needle_is_grasp = False
            self.grasp_steps = 0

        """" Calculates the averages of the metrics """
        self._average_psm_needle_dist.append(_psm2_to_Needle)
        self._average_needle_entry_dist.append(self.needle_entry_dist)

        self.AVG_needle_entry_dist = self.Average(self._average_needle_entry_dist)
        self.AVG_psm_needle_dist = self.Average(self._average_psm_needle_dist)

        #print(f'Metrics:')
        #print(f'avg_psm_needle {self.AVG_needle_entry_dist} | avg_needle_entry {self.AVG_psm_needle_dist} ')
        #print(f'grasp_steps {self.grasp_steps} | needle_entry_dist {self.needle_entry_dist} | _psm2_to_Needle {_psm2_to_Needle} ')


    def _get_psm2_ik(self):
        pos_psm2 = self.psm2.measured_jp().pose
        current_state = PyKDL.Frame(
            PyKDL.Rotation.Quaternion(pos_psm2.orientation.x, pos_psm2.orientation.y, pos_psm2.orientation.z,
                                      pos_psm2.orientation.w),
            PyKDL.Vector(pos_psm2.position.x, pos_psm2.position.y, pos_psm2.position.z))
        current_state_js = psm_ik.compute_IK(current_state)
        return current_state_js

    def _set_reached_frame(self, target_state_js, current_state_js):
        delta = np.array(target_state_js, dtype=np.float32) - np.array(current_state_js, dtype=np.float32)
        delta_norm = np.linalg.norm(delta)

        if abs(delta_norm - self.prev_delta_norm) <= 0.00005: #0.0001:
            self._reached_frame_flag = True
        else:
            self._reached_frame_flag = False
        self.prev_delta_norm = delta_norm

    def internal_action(self, action):
        # Performs linear interpolation from current position to Oracle target position
        current_state_js = self._get_psm2_ik()
        target_state_js = psm_ik.compute_IK(action)

        # Compute the step size
        step_size = 0.05
        delta = np.array(target_state_js, dtype=np.float32) - np.array(current_state_js, dtype=np.float32)
        delta_norm = np.linalg.norm(delta)
        if delta_norm > step_size:  #TODO: Not sure this logic is correct, check
           step = delta * (step_size / delta_norm)
        else:
           step = delta

        next_action = np.array(current_state_js, dtype=np.float32) + step
        self.psm2.servo_jp(next_action)
        rospy.sleep(0.05)  # 20 Hz control signal

        # Update the reached flag
        current_state_js = self._get_psm2_ik()
        self._set_reached_frame(target_state_js, current_state_js) #TODO: Seems abit redundant, could be optimized

        if self.Random_Needle_drop and (random.randint(0, 100) == 1) and not self.Has_dropped_once and (self.SM_STATE == "ABOVE_NEEDLE" or self.SM_STATE == "GOTO_ENTRY" or self.SM_STATE == "NEEDLE_INSERTION" ):
            print(f'{self.SM_STATE}: Dropping Needle, Ooops')
            self.move_jaw('psm2', 0.8)
            time.sleep(1.0)
            self.move_jaw('psm2', 0.0)
            time.sleep(1.0)
            self.Has_dropped_once = True

    def _set_action(self):
        STATE = self.SM_STATE
        def convert_pose_to_action(pose, gripper_angle):
            px, py, pz, ox, oy, oz, ow = pose
            return [px, py, pz, ox, oy, oz, ow, gripper_angle]

        if STATE == "GOTO_NEEDLE":
            self.t_base_grasp = self.tf_buffer.lookup_pose('psm2/baselink', 'Needle/Grasp')
            self.t_above_needle = pos_to_action(self.t_base_grasp @ Pose(position=[0.0, 0, -0.01]))
            action = convert_pose_to_action(self.t_above_needle, self.init_grasp)
        elif STATE == "GRASP_NEEDLE_1":
            action = convert_pose_to_action(self.t_grasp_needle, self.init_grasp)
        elif STATE == "GRASP_NEEDLE_2":
            action = convert_pose_to_action(self.t_grasp_needle, 0)
        elif STATE == "ABOVE_NEEDLE":
            #action = convert_pose_to_action(self.t_above_needle, 0)
            action = convert_pose_to_action(self.t_grasp_needle, 0)
        elif STATE == "GOTO_ENTRY":
            action = convert_pose_to_action(self.t_base_tcp_desired, 0)
        elif STATE == "NEEDLE_INSERTION":
            action = convert_pose_to_action(self.t_base_tcp_desired, 0)
        elif STATE == "DONE":
            action = convert_pose_to_action(self.t_base_tcp_desired, 0)

        self.current_action = action

    def _update_states(self, reached_frame, verbose=0):

        # Compute the next state
        if self.SM_STATE == "GOTO_NEEDLE" and reached_frame:
            self.SM_STATE = "GRASP_NEEDLE_1"
            self.t_base_grasp = self.tf_buffer.lookup_pose('psm2/baselink', 'Needle/Grasp')
            self.t_above_needle = pos_to_action(self.t_base_grasp @ Pose(position=[0, 0, -0.01]))
            self.t_grasp_needle = pos_to_action(self.t_base_grasp @ Pose(position=[0.0, 0, 0.0045]))

        elif self.SM_STATE == "GRASP_NEEDLE_1" and reached_frame:
            self.SM_STATE = "GRASP_NEEDLE_2"
        elif self.SM_STATE == "GRASP_NEEDLE_2" and reached_frame:
            self.SM_STATE = "ABOVE_NEEDLE"
            if not self.needle_is_grasp:
                self.SM_STATE = "GOTO_NEEDLE"

        elif self.SM_STATE == "ABOVE_NEEDLE" and reached_frame:
            self.t_tcp0_needletip = Pose.from_msg(self.psm2.msg_measured_cp.pose).inverse() @ self.tf_buffer.lookup_pose(
                'psm2/baselink', 'Needle/Tip')
            self.t_base_tcp_desired = pos_to_action(
                self.tf_buffer.lookup_pose('psm2/baselink', self.entry_target_string) @ Pose.from_axis_angle([1, 0, 0],
                -np.pi / 2) @ self.t_tcp0_needletip.inverse())
            self.SM_STATE = "GOTO_ENTRY"
            if not self.needle_is_grasp:
                self.SM_STATE = "GOTO_NEEDLE"

        elif self.SM_STATE == "GOTO_ENTRY" and reached_frame:
            self.SM_STATE = "NEEDLE_INSERTION"
            t_tcp0_needle = Pose.from_msg(self.psm2.msg_measured_cp.pose).inverse() @ self.tf_buffer.lookup_pose('psm2/baselink','Needle')
            t_base_needle = self.tf_buffer.lookup_pose('psm2/baselink', 'Needle')
            self.t_base_tcp_desired = pos_to_action(t_base_needle @ Pose.from_axis_angle([0, 0, 1], -np.pi/2) @ t_tcp0_needle.inverse())

            if not self.needle_is_grasp:
                self.SM_STATE = "GOTO_NEEDLE"

        elif self.SM_STATE == "NEEDLE_INSERTION" and reached_frame:
            self.SM_STATE = "DONE"
            if not self.needle_is_grasp:
                self.SM_STATE = "GOTO_NEEDLE"

        elif self.SM_STATE == "DONE":
            self.done = False #True # TODO: set this to true to enable finish episode
            # if not self.done:
            #     self.SM_STATE = "GOTO_ENTRY"

        if verbose >= 1:
            print(f"Reached target, next target: {self.SM_STATE}")

    def get_Oracle_action(self):
        # Updating the state machine for the suture oracle
        self._update_states(self._reached_frame_flag, 1)

        # Based on the state set the high level action
        self._set_action()

        if not self.Random_Needle_drop:
            self.Random_Needle_drop = False #True
            #print(f'Activating Random_Needle_drop: {self.Random_Needle_drop}')
        return self.current_action

    def set_target_entry(self, entry_num):
        if entry_num == 1:
            entry_pose = self.scene.measured_cp(SceneObjectType.Entry1).pose
        if entry_num == 2:
            entry_pose = self.scene.measured_cp(SceneObjectType.Entry2).pose
        if entry_num == 3:
            entry_pose = self.scene.measured_cp(SceneObjectType.Entry3).pose
        if entry_num == 4:
            entry_pose = self.scene.measured_cp(SceneObjectType.Entry4).pose

        self.pos_entry_target_number = entry_num

        print(f'Target entry has been set to: Entry{entry_num}')
        self.entry_target_string = f'Entry{entry_num}'
        self.pos_entry = [entry_pose.position.x, entry_pose.position.y, entry_pose.position.z, entry_pose.orientation.x, entry_pose.orientation.y, entry_pose.orientation.z, entry_pose.orientation.w]

    def reset_world(self, done):
        print("Resetting the world")
        self.world_handle.reset()
        time.sleep(0.5)

        # If the simulation is done, then don't move the needle
        if not done:
            self.set_random_needle()

        #self.world_handle.reset_bodies()
        self.move_jaw('psm1', 0.8)
        self.move_jaw('psm2', 0.8)

        T_e_b = Frame(Rotation.RPY(np.pi, 0, np.pi / 2.), Vector(0., 0., -0.13))
        self.move_to_frame(arm_name='psm1', frame=T_e_b, verbose=0)

        T_e_b = Frame(Rotation.RPY(np.pi, 0, np.pi / 4.), Vector(0.01, -0.01, -0.13))
        self.move_to_frame(arm_name='psm2', frame=T_e_b, verbose=0)
        time.sleep(0.5)

        self.SM_STATE = "GOTO_NEEDLE"
        self.done = False

        # Reset metrics
        self.needle_entry_dist = 999
        #self.accumulated_optimal_action_error_norm = 0
        self.grasp_steps = 0
        self.pickup_attempts = 0
        self.init_grasp = random.uniform(0.5, 1.0)

        self._average_needle_entry_dist = []
        self._average_psm_needle_dist = []

        self.Random_Needle_drop = False
        self.Has_dropped_once = False
        #print(f'Resetting Random_Needle_drop: {self.Random_Needle_drop}')
        time.sleep(0.5)

    def send_action(self, action):
        frame = Frame(
            Rotation.Quaternion(action[3], action[4], action[5], action[6]),
            Vector(action[0], action[1], action[2]),
        )
        self.internal_action(frame)

        #if self.needle_entry_dist <= 0.0008 and self.needle_is_grasp: #TODO: Chould this be removed?
        #    self.done = True

    def move_to_frame(self, arm_name, frame, verbose=0):
        if arm_name == "psm1":
            self.psm1.servo_cp(frame)
        if arm_name == "psm2":
            self.psm2.servo_cp(frame)
        if verbose!=0:
            print(f'Setting the end-effector frame of {arm_name} w.r.t Base', frame)
    def move_to_joint_pos(self, arm_name, joint_position, verbose=0):
        if arm_name == "psm1":
            self.psm1.servo_jp(joint_position)
        if arm_name == "psm2":
            self.psm2.servo_jp(joint_position)
        if verbose!=0:
            print(f'Setting the end-effector frame of {arm_name} w.r.t Base', joint_position)

    def move_jaw(self, arm_name, angle):
        # Close the jaw of the surgical arm to grasp the needle
        if arm_name == "psm1":
            if angle != self._psm1_angle:
                for i in np.linspace(self._psm1_angle, angle, num=20):
                    self.psm1.set_jaw_angle(i)
                    rospy.sleep(0.1)
            self._psm1_angle = angle
        if arm_name == "psm2":
            if angle != self._psm2_angle:
                for i in np.linspace(self._psm2_angle, angle, num=20):
                    self.psm2.set_jaw_angle(i)
                    rospy.sleep(0.1)
            self._psm2_angle = angle

    def get_observation(self):
        # TODO: could also include joint angles, velocity and acceleration
        # Get observation returns the current observation space

        # PSM arms states
        cp_1 = self.psm1.measured_jp().pose
        pos_psm1 = [cp_1.position.x, cp_1.position.y, cp_1.position.z, cp_1.orientation.x, cp_1.orientation.y, cp_1.orientation.z, cp_1.orientation.w]
        cp_2 = self.psm2.measured_jp().pose
        pos_psm2 = [cp_2.position.x, cp_2.position.y, cp_2.position.z, cp_2.orientation.x, cp_2.orientation.y, cp_2.orientation.z, cp_2.orientation.w]

        # Needle pose
        needle = self.scene.measured_cp(SceneObjectType.Needle).pose
        pos_needle = [needle.position.x, needle.position.y, needle.position.z, needle.orientation.x, needle.orientation.y, needle.orientation.z, needle.orientation.w]

        # TODO: implement vision pipeline interface
        # get image data
        #image = self.saver.save_image_data('mono_left')
        #pos_needle_EST = self.vision_method.getEstimation(image, "regression")
        #reduced_pos_needle_EST = pos_needle_EST[:3]
        #Needle_est_error = np.array(pos_needle) - np.array(pos_needle_EST)

        #print(f'Error: xyz {np.linalg.norm(Needle_est_error[:3])}, quat {sum(abs(Needle_est_error[3:7]))}') # TODO: calculate the quat rot error
        #print(f"{pos_needle} vs. {pos_needle_EST} vs. {Needle_est_error} ")
        #print(f" {pos_needle} vs. {pos_needle_EST}")

        # open the file in the write mode
        # with open("csv_files/NeedleError.csv", 'a') as f:
        #     # datetime object containing current date and time
        #     now = datetime.now()
        #     dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        #     # create the csv writer
        #     writer = csv.writer(f)
        #     # Add the data in row
        #     row = [dt_string, pos_needle, pos_needle_EST, Needle_est_error]
        #     # write a row to the csv file
        #     writer.writerow(row)

        # add noise to needle
        noisy_pos_needle = np.array(pos_needle, dtype=np.float32)
        #print(f'Before noise: {noisy_pos_needle}')
        mu, sigma = 0, 0.001  # mean and standard deviation
        noise = np.random.normal(mu, sigma, 7)
        noisy_pos_needle = noisy_pos_needle + noise
        #print(f'After noise: {noisy_pos_needle}')

        # Get target entry point
        pos_entry_target = self.pos_entry
        pos_entry_target_number = [self.pos_entry_target_number]

        # state = {
        #     #'psm1': np.array(pos_psm1, dtype=np.float32),
        #     'psm2': np.array(pos_psm2, dtype=np.float32),
        #     #'needle': np.array(pos_needle, dtype=np.float32),              # Simulator needle pose
        #     #'needle': np.array(pos_needle_EST, dtype=np.float32),          # Estimated needle pose
        #     #'needle': np.array(reduced_pos_needle_EST, dtype=np.float32),           # Reduced needle pose
        #     'needle': noisy_pos_needle, # Artificial noisy data
        #     #'entry': np.array(pos_entry_target, dtype=np.float32),
        #     'entry': np.array(pos_entry_target_number, dtype=np.float32)
        #     #'image': np.array(image, dtype=np.float32)
        # }

        image = self.saver.save_image_data('mono_left')
        state_vector = self.vision_method.getEstimation(image, "AE")
        state = {'state': np.array(state_vector, dtype=np.float32)}

        return state

    def get_vision_data(self):
        # get image data
        image = self.saver.save_image_data('mono_left')
        # Needle pose
        needle = self.scene.measured_cp(SceneObjectType.Needle).pose
        pos_needle = [needle.position.x, needle.position.y, needle.position.z, needle.orientation.x,
                      needle.orientation.y, needle.orientation.z, needle.orientation.w]
        return image, pos_needle

    def get_video_feed(self):
        # Stereo image from ecm
        print(f'Added new frame')
        image = self.saver.get_frame()
        return image

    def get_reward(self, done):
        # TODO: Define reward function
        if self.needle_entry_dist <= 0.0008 and self.needle_is_grasp:
            print(f'Task Succesfully completed, needle_entry_dist: {self.needle_entry_dist}, needle_is_grasp: {self.needle_is_grasp}')
            #self.done = True
        #else:
            #print(f'Task failed, needle_entry_dist: {self.needle_entry_dist}, needle_is_grasp: {self.needle_is_grasp}')

        reward = 0
        if self.needle_entry_dist <= 100:
            if self.needle_is_grasp:
                reward = 1 - self.needle_entry_dist*10
                #reward = self.needle_entry_dist

        print(f'reward: {reward}')
        return reward



