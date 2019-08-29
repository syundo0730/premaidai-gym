import os
from math import sqrt, atan2, sin, cos

import numpy as np
from roboschool.scene_abstract import cpp_household
from roboschool.gym_urdf_robot_env import RoboschoolUrdfEnv
from roboschool.multiplayer import SharedMemoryClientEnv
from roboschool.scene_stadium import SinglePlayerStadiumScene


class RoboschoolPremaidAIEnv(SharedMemoryClientEnv, RoboschoolUrdfEnv):
    JOINT_DIM = 25
    # joint position & speed => JOINT_DIM * 2
    # body roll, pitch, sin(yaw), cos(yaw) => 4
    # rpy speed, acc_x, acc_y, acc_z => 6
    # target body height diff => 1
    OBS_DIM = JOINT_DIM * 2 + 4 + 6 + 1

    def __init__(self):
        model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'assets/premaidai.urdf')
        RoboschoolUrdfEnv.__init__(self, model_urdf=model_path, robot_name='base_link',
                                   action_dim=self.JOINT_DIM, obs_dim=self.OBS_DIM,
                                   fixed_base=False, self_collision=False)
        self.rewards = []
        self._last_body_rpy = None
        self._last_body_speed = None
        self._walk_target_distance = 0
        self._walk_target_yaw = 0
        self._target_xyz = (1e3, 0, 0.2)  # kilometer away

    def create_single_player_scene(self):
        return SinglePlayerStadiumScene(gravity=9.8, timestep=0.0165 / 8, frame_skip=8)

    def robot_specific_reset(self):
        cpose = cpp_household.Pose()
        cpose.set_xyz(0, 0, 0.26)
        cpose.set_rpy(0, 0, 0)
        self.cpp_robot.set_pose_and_speed(cpose, 0, 0, 0)

        for j in self.ordered_joints:
            j.reset_current_position(0, 0)
        self.scene.actor_introduce(self)

    def calc_state(self):
        robot_pose = self.robot_body.pose()
        joint_angles_and_speeds = np.array(
            [j.current_position() for j in self.ordered_joints], dtype=np.float32).flatten()
        body_rpy = robot_pose.rpy()
        roll, pitch, yaw = body_rpy
        body_speed = self.robot_body.speed()
        dt = self.scene.dt
        body_rpy_speed = [(rpy - last_rpy) / dt for rpy, last_rpy in zip(body_rpy, self._last_body_rpy)] \
            if self._last_body_rpy else [0., 0., 0.]
        body_acc = [(speed - last_speed) / dt for speed, last_speed in zip(body_speed, self._last_body_speed)] \
            if self._last_body_speed else [0., 0., 0.]

        x, y, z = robot_pose.xyz()
        target_x, target_y, target_z = self._target_xyz
        diff_x, diff_y = target_x - x, target_y - y
        self._walk_target_distance = sqrt(diff_x**2 + diff_y**2)
        delta_angle_to_target = atan2(diff_y, diff_x) - yaw
        self._last_body_rpy = body_rpy
        self._last_body_speed = body_speed
        return np.concatenate([joint_angles_and_speeds,
                               [roll, pitch, cos(delta_angle_to_target), sin(delta_angle_to_target)],
                               body_rpy_speed, body_acc, [z - target_z]])

    def step(self, action):
        self._apply_action(action)
        self.scene.global_step()
        state = self.calc_state()
        reward = self._calc_reward(state, action)
        done = self._is_done(state)

        # === for debug UI ===
        self.frame += 1
        self.reward = reward
        self.done = done
        self.HUD(state, action, done)
        # ====================

        return state, reward, done, {}

    def _apply_action(self, action):
        for n, j in enumerate(self.ordered_joints):
            j.set_servo_target(float(action[n]), 0.1, 0.1, 40)

    def _is_done(self, state):
        raise NotImplementedError

    def _calc_reward(self, state, action):
        raise NotImplementedError


class RoboschoolPremaidAIWalker(RoboschoolPremaidAIEnv):
    ELECTRICITY_COST_WEIGHT = -2.0
    STALL_TORQUE_COST_WEIGHT = -0.1

    def __init__(self):
        super().__init__()
        self._last_potential = None
        self._last_joint_speed = None

    def _is_done(self, state):
        _, _, z = self.robot_body.pose().xyz()
        # prevent fallen down and jumping
        return z < 0.1 or z > 0.4

    def _calc_reward(self, state, action):
        # calculate potential cost
        dt = self.scene.dt
        potential = - self._walk_target_distance / dt
        progress = potential - self._last_potential if self._last_potential else 0
        self._last_potential = potential

        # calculate joint cost
        joint_speed = state[1::2][:self.JOINT_DIM]
        joint_acc = ((joint_speed - self._last_joint_speed) / dt
                     if self._last_joint_speed else np.zeros_like(joint_speed))
        electricity_cost = (self.ELECTRICITY_COST_WEIGHT * float(np.abs(joint_acc * joint_speed).mean()) +
                            self.STALL_TORQUE_COST_WEIGHT * float(np.square(joint_acc).mean()))

        # calculate standing cost
        height_cost = -abs(self.robot_body.pose().xyz()[2] - self._target_xyz[2])

        self.rewards = [
            progress,
            electricity_cost,
            height_cost,
        ]
        return sum(self.rewards)
