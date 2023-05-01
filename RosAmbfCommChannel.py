
# Import the relevant classes
import rospy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from PyKDL import Frame, Rotation, Vector
import time
from enum import Enum
import numpy as np
from std_msgs.msg import Empty
from sensor_msgs.msg import Image
from surgical_robotics_challenge.task_completion_report import TaskCompletionReport

import collections
from gym import spaces


# Connect to AMBF and setup image suscriber
import img_saver as ImgSaver

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
        self.cp_pub = rospy.Publisher(arm_name + "/servo_cp", PoseStamped, queue_size=1)
        self.jp_pub = rospy.Publisher(arm_name + "/servo_jp", JointState, queue_size=1)
        self.jaw_jp_pub = rospy.Publisher(arm_name + '/jaw/' + 'servo_jp', JointState, queue_size=1)

        self.measured_cp_msg = None
        self.T_b_w_msg = None
        self.measured_jp_msg = None

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

    def servo_jp(self, jp):
        if type(jp) == list:
            msg = list_to_sensor_msg_position(jp)
        else:
            msg = jp
        self.jp_pub.publish(msg)

    def set_jaw_angle(self, val):
        msg = list_to_sensor_msg_position([val])
        self.jaw_jp_pub.publish(msg)


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

        namespace = '/CRTK/'
        suffix = '/measured_cp'
        for k, i in self._scene_object_poses.items():
            self._subs.append(rospy.Subscriber(namespace + k.name + suffix, PoseStamped,
                                               self.state_cb, callback_args=k, queue_size=1))

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
    def __init__(self):
        self.saver = ImgSaver.ImageSaver()
        # Create an instance of the client
        rospy.init_node('RosAmbfCC')
        time.sleep(0.5)
        self.world_handle = WorldInterface()

        # Get a handle to PSM1
        self.psm1 = ARMInterface(ArmType.PSM1)
        # Get a handle  to PSM2
        self.psm2 = ARMInterface(ArmType.PSM2)
        # Get a handle to ECM
        self.ecm = ARMInterface(ArmType.ECM)
        # Get a handle to scene to access its elements, i.e. needle and entry / exit points
        self.scene = SceneInterface()
        # Create an instance of task completion report with you team name
        #task_report = TaskCompletionReport(team_name='MT_SDU_IBC')
        # Small sleep to let the handles initialize properly
        add_break(0.5)

        # Add you camera stream subs
        self.cameraL_sub = ImageSub('/ambf/env/cameras/cameraL/ImageData')
        self.cameraR_sub = ImageSub('/ambf/env/cameras/cameraR/ImageData')

        # Set entry target
        self.pos_entry = None
        self.set_target_entry(1)

    def set_target_entry(self, entry_num):
        if entry_num == 1:
            entry_pose = self.scene.measured_cp(SceneObjectType.Entry1).pose
        if entry_num == 2:
            entry_pose = self.scene.measured_cp(SceneObjectType.Entry2).pose
        if entry_num == 3:
            entry_pose = self.scene.measured_cp(SceneObjectType.Entry3).pose
        if entry_num == 4:
            entry_pose = self.scene.measured_cp(SceneObjectType.Entry4).pose

        print(f'Target entry has been set to: Entry{entry_num}')

        self.pos_entry = [entry_pose.position.x, entry_pose.position.y, entry_pose.position.z, entry_pose.orientation.x, entry_pose.orientation.y, entry_pose.orientation.z, entry_pose.orientation.w]

    def reset_world(self):
        print("Resetting the world")
        self.world_handle.reset()
        self.move_jaw('psm1', 0.8)
        self.move_jaw('psm2', 0.8)

        T_e_b = Frame(Rotation.RPY(np.pi, 0, np.pi / 2.), Vector(0., 0., -0.13))
        self.move_to_frame(arm_name='psm1', frame=T_e_b, verbose=0)

        T_e_b = Frame(Rotation.RPY(np.pi, 0, np.pi / 4.), Vector(0.01, -0.01, -0.13))
        self.move_to_frame(arm_name='psm2', frame=T_e_b, verbose=0)

        time.sleep(3)

    def send_action(self, action):
        frame = Frame(
            Rotation.Quaternion(action[3], action[4], action[5], action[6]),
            Vector(action[0], action[1], action[2]),
        )
        self.move_to_frame('psm2',frame, 1)
        time.sleep(0.1)
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
        if arm_name == "psm1":
            self.psm1.set_jaw_angle(angle)
        if arm_name == "psm2":
            self.psm2.set_jaw_angle(angle)
    # def get_observation(self):
    #     # TODO: could also include joint angles, velocity and acceleration
    #     # Get observation returns the current observation space
    #
    #     # PSM arms states
    #     cp_1 = self.psm1.measured_jp().pose
    #     pos_psm1 = {
    #         "position": [cp_1.position.x, cp_1.position.y, cp_1.position.z],
    #         "orientation": [cp_1.orientation.x, cp_1.orientation.y, cp_1.orientation.z, cp_1.orientation.w]
    #     }
    #     cp_2 = self.psm2.measured_jp().pose
    #     pos_psm2 = {
    #         "position": [cp_2.position.x, cp_2.position.y, cp_2.position.z],
    #         "orientation": [cp_2.orientation.x, cp_2.orientation.y, cp_2.orientation.z, cp_2.orientation.w]
    #     }
    #
    #     # Needle pose
    #     needle = self.scene.measured_cp(SceneObjectType.Needle).pose
    #     pos_needle = {
    #         "position": [needle.position.x, needle.position.y, needle.position.z],
    #         "orientation": [needle.orientation.x, needle.orientation.y, needle.orientation.z, needle.orientation.w]
    #     }
    #
    #     # Get target entry point
    #     pos_entry_target = self.pos_entry
    #
    #     # Stereo image from ecm
    #     image = self.saver.save_image_data()
    #
    #     # Format the observation space
    #     observation = {
    #         "arm1": pos_psm1,
    #         "arm2": pos_psm2,
    #         "needle": pos_needle,
    #         "entry": pos_entry_target,
    #         "image": image
    #     }
    #
    #     return observation

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

        # Get target entry point
        pos_entry_target = self.pos_entry

        # Stereo image from ecm
        image = self.saver.save_image_data()

        # obs_dict = collections.OrderedDict(
        #     psm1=spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32),
        #     psm2=spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32),
        #     needle=spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32),
        #     entry=spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32),
        #     image=spaces.Box(low=0.0, high=255.0, shape=(960, 540, 3), dtype=np.float32)
        # )
        #
        # # Fill in the values for each observation space
        # obs_dict['psm1'] = np.array(pos_psm1, dtype=np.float32)
        # obs_dict['psm2'] = np.array(pos_psm2, dtype=np.float32)
        # obs_dict['needle'] = np.array(pos_needle, dtype=np.float32)
        # obs_dict['entry'] = np.array(pos_entry_target, dtype=np.float32)
        # obs_dict['image'] = np.array(image, dtype=np.float32)

        state = {
            'psm1': np.array(pos_psm1, dtype=np.float32),
            'psm2': np.array(pos_psm2, dtype=np.float32),
            'needle': np.array(pos_needle, dtype=np.float32),
            'entry': np.array(pos_entry_target, dtype=np.float32),
            #'image': np.array(image, dtype=np.float32)
        }

        #return spaces.Dict(obs_dict)
        #return obs_dict
        return state

    def get_reward(self, done):
        # TODO: Define reward function
        return 0

