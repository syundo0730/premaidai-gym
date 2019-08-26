import os
from math import sqrt

import numpy as np
from roboschool.scene_abstract import cpp_household
from roboschool.gym_urdf_robot_env import RoboschoolUrdfEnv
from roboschool.multiplayer import SharedMemoryClientEnv
from roboschool.scene_stadium import SinglePlayerStadiumScene


class RoboschoolPremaidAIEnv(SharedMemoryClientEnv, RoboschoolUrdfEnv):
    JOINT_DIM = 25
    OBS_DIM = JOINT_DIM * 2 + 9  # joint position & speed, body rpy(roll, pitch, yaw) & rpy speed, acc_x, acc_y, acc_z

    def __init__(self):
        model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'assets/premaidai.urdf')
        RoboschoolUrdfEnv.__init__(self, model_urdf=model_path, robot_name='base_link',
                                   action_dim=self.JOINT_DIM, obs_dim=self.OBS_DIM,
                                   fixed_base=False, self_collision=False)
        self.rewards = []
        self._last_body_rpy = None
        self._last_body_speed = None

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
        body_speed = self.robot_body.speed()
        dt = self.scene.dt
        body_rpy_speed = [(rpy - last_rpy) / dt for rpy, last_rpy in zip(body_rpy, self._last_body_rpy)] \
            if self._last_body_rpy else [0., 0., 0.]
        body_acc = [(speed - last_speed) / dt for speed, last_speed in zip(body_speed, self._last_body_speed)] \
            if self._last_body_speed else [0., 0., 0.]
        self._last_body_rpy = body_rpy
        self._last_body_speed = body_speed
        return np.concatenate([joint_angles_and_speeds, body_rpy, body_rpy_speed, body_acc])

    def step(self, action):
        self._apply_action(action)
        self.scene.global_step()
        state = self.calc_state()
        reward = self._calc_reward(state)
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
            j.set_servo_target(action[n], 0.1, 0.1, 40)

    def _is_done(self, state):
        raise NotImplementedError

    def _calc_reward(self, state):
        raise NotImplementedError


class RoboschoolPremaidAIWalker(RoboschoolPremaidAIEnv):
    def __init__(self):
        super().__init__()
        self._target_xy = (1e3, 0)  # kilometer away
        self._last_potential = None

    def _is_done(self, state):
        _, _, z = self.robot_body.pose().xyz()
        return z < 0.1

    def _calc_reward(self, state):
        x, y, _ = self.robot_body.pose().xyz()
        target_x, target_y = self._target_xy
        diff_x, diff_y = target_x - x, target_y - y
        potential = - sqrt((diff_x**2 + diff_y**2)) / self.scene.dt
        progress = potential - self._last_potential if self._last_potential else 0
        self._last_potential = potential
        self.rewards = [progress]
        return progress
