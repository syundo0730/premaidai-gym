import os
from math import sqrt, atan2, radians

import numpy as np
from roboschool.scene_abstract import cpp_household
from roboschool.gym_urdf_robot_env import RoboschoolUrdfEnv
from roboschool.multiplayer import SharedMemoryClientEnv
from roboschool.scene_stadium import SinglePlayerStadiumScene


class RoboschoolPremaidAIEnv(SharedMemoryClientEnv, RoboschoolUrdfEnv):
    JOINT_DIM = 25
    # joint position & speed => JOINT_DIM * 2
    # body roll, pitch, delta_angle_to_target => 3
    # rpy speed, acc_x, acc_y, acc_z => 6
    # target body height diff => 1
    OBS_DIM = JOINT_DIM * 2 + 3 + 6 + 1
    FOOT_NAME_LIST = ["r_foot", "l_foot"]
    HOME_POSITION = np.array(
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, radians(60), 0, 0, 0, 0, radians(-60), 0, 0, 0, 0, 0, 0])

    def __init__(self):
        model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'assets/premaidai.urdf')
        RoboschoolUrdfEnv.__init__(self, model_urdf=model_path, robot_name='base_link',
                                   action_dim=self.JOINT_DIM, obs_dim=self.OBS_DIM,
                                   fixed_base=False, self_collision=False)
        self._last_camera_x = 0
        self.rewards = []
        self._last_body_rpy = None
        self._last_body_speed = None
        self._walk_target_distance = 0
        self._walk_target_yaw = 0
        self._target_xyz = (1e3, 0, 0.2)  # kilometer away
        self._feet_objects = []

    def create_single_player_scene(self):
        return SinglePlayerStadiumScene(gravity=9.8, timestep=0.0165 / 8, frame_skip=8)

    def robot_specific_reset(self):
        for i, j in enumerate(self.ordered_joints):
            j.reset_current_position(self.HOME_POSITION[i], 0)
            limits = j.limits()
            # should set up low, high at here because orderd_joints are set in reset method
            self.action_space.low[i] = limits[0]
            self.action_space.high[i] = limits[1]
        self._feet_objects = [self.parts[name] for name in self.FOOT_NAME_LIST]
        self.scene.actor_introduce(self)
        cpose = cpp_household.Pose()
        cpose.set_xyz(0, 0, 0.27)
        cpose.set_rpy(0, 0, 0)
        self.cpp_robot.set_pose_and_speed(cpose, 0, 0, 0)

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
                               [roll, pitch, delta_angle_to_target],
                               body_rpy_speed, body_acc, [z - target_z]])

    def camera_adjust(self):
        # simple follow
        x, y, z = self.robot_body.pose().xyz()
        camera_x = 0.98 * self._last_camera_x + (1 - 0.98)*x
        self.camera.move_and_look_at(camera_x, y - 2.0, 1.4, x, y, 1.0)
        self._last_camera_x = camera_x

    def step(self, action):
        if self.frame < 1:
            # just apply same action as reset to stabilize simulation on 1st step
            action = self.HOME_POSITION
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
            j.set_servo_target(float(action[n]), 0.045, 0.045, 1.372)

    def _is_done(self, state):
        raise NotImplementedError

    def _calc_reward(self, state, action):
        raise NotImplementedError


class RoboschoolPremaidAIWalker(RoboschoolPremaidAIEnv):
    PROGRESS_COST_WEIGHT = 10.
    ELECTRICITY_COST_WEIGHT = -2.
    STALL_TORQUE_COST_WEIGHT = -0.1
    JOINT_POWER_COEF = 0.01
    JOINT_AT_LIMIT_COST_WEIGHT = -0.2
    FOOT_SELF_COLLISION_COST_WEIGHT = -1.
    FOOT_SELF_COLLISION_EXCEPTION = {
        'r_foot': {'r_ankle', 'r_lleg'},
        'l_foot': {'l_ankle', 'l_lleg'},
    }

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
        progress *= self.PROGRESS_COST_WEIGHT
        self._last_potential = potential

        # calculate joint cost
        joint_speed = state[1::2][:self.JOINT_DIM]
        joint_acc = ((joint_speed - self._last_joint_speed) / dt
                     if self._last_joint_speed is not None else np.zeros_like(joint_speed))
        joint_acc *= self.JOINT_POWER_COEF
        self._last_joint_speed = joint_speed
        electricity_cost = (self.ELECTRICITY_COST_WEIGHT * np.abs(joint_acc * joint_speed).mean() +
                            self.STALL_TORQUE_COST_WEIGHT * np.linalg.norm(joint_acc))

        # joint at limit cost
        relative_angles = np.array([j.current_relative_position()[0] for j in self.ordered_joints], dtype=np.float32)
        joints_at_limit_count = np.count_nonzero(np.abs(relative_angles[0::2]) > 0.92)
        joints_at_limit_cost = self.JOINT_AT_LIMIT_COST_WEIGHT * joints_at_limit_count

        # calculate height cost
        _, _, z = self.robot_body.pose().xyz()
        height_cost = 2. if z > 0.15 else -1.

        # calculate feet self collision cost
        collisions = [(f.name, part.name) for f in self._feet_objects for part in self.cpp_robot.parts
                      if part.name != f.name
                      and part.name not in self.FOOT_SELF_COLLISION_EXCEPTION[f.name]
                      and np.linalg.norm(np.array(f.pose().xyz()) - np.array(part.pose().xyz())) < 0.06]
        # print(collisions)
        feet_collision_cost = self.FOOT_SELF_COLLISION_COST_WEIGHT * len(collisions)

        self.rewards = [
            progress,
            electricity_cost,
            joints_at_limit_cost,
            height_cost,
            feet_collision_cost,
        ]
        return sum(self.rewards)
