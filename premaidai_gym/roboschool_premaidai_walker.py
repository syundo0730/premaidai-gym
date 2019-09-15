import os
import random
from math import sqrt, atan2, radians, cos, sin, exp

import numpy as np
from roboschool.scene_abstract import cpp_household
from roboschool.gym_urdf_robot_env import RoboschoolUrdfEnv
from roboschool.multiplayer import SharedMemoryClientEnv
from roboschool.scene_stadium import SinglePlayerStadiumScene

from premaidai_gym.simple_walk_controller import SimpleWalkController


class RoboschoolPremaidAIEnv(SharedMemoryClientEnv, RoboschoolUrdfEnv):
    JOINT_DIM = 25
    FOOT_NAME_LIST = ["r_foot", "l_foot"]
    HOME_POSITION = np.array(
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, radians(60), 0, 0, 0, 0, radians(-60), 0, 0, 0, 0, 0, 0])

    # these values are brought from KRS2552HV spec, servo motor that premaidAI is actually using for leg joints
    JOINT_MAX_TORQUE = 1.372  # 14.0kgf * cm
    JOINT_MAX_SPEED = radians(428.6)  # 0.14s / 60deg

    def __init__(self, action_dim, obs_dim):
        model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'assets/premaidai.urdf')
        RoboschoolUrdfEnv.__init__(self, model_urdf=model_path, robot_name='base_link',
                                   action_dim=action_dim, obs_dim=obs_dim,
                                   fixed_base=False, self_collision=False)
        self._last_camera_x = 0
        self.rewards = []
        self._last_body_rpy = None
        self._last_body_speed = None
        self._walk_target_distance = 0
        self._walk_target_yaw = 0
        self._target_xyz = (1e3, 0, 0.25)  # kilometer away
        self._feet_objects = []
        self._head = None

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
        cpose.set_rpy(0, 0, random.uniform(-radians(45), radians(45)))
        self.cpp_robot.set_pose_and_speed(cpose, 0, 0, 0)
        self._head = self.parts['head']

    def calc_state(self):
        robot_pose = self.robot_body.pose()
        joint_angles_and_speeds = np.array(
            [j.current_relative_position() for j in self.ordered_joints], dtype=np.float32).flatten()
        body_rpy = robot_pose.rpy()
        roll, pitch, yaw = body_rpy
        body_speed = self.robot_body.speed()
        dt = self.scene.dt
        body_rpy_speed = np.array([(rpy - last_rpy) / dt for rpy, last_rpy in zip(body_rpy, self._last_body_rpy)]
                                  if self._last_body_rpy else [0., 0., 0.])
        body_acc = np.array([(speed - last_speed) / dt for speed, last_speed in zip(body_speed, self._last_body_speed)]
                            if self._last_body_speed else [0., 0., 0.])

        x, y, z = robot_pose.xyz()
        target_x, target_y, target_z = self._target_xyz
        diff_x, diff_y = target_x - x, target_y - y
        self._walk_target_distance = sqrt(diff_x**2 + diff_y**2)
        delta_angle_to_target = atan2(diff_y, diff_x) - yaw
        self._last_body_rpy = body_rpy
        self._last_body_speed = body_speed
        return np.clip(np.concatenate(
            [joint_angles_and_speeds,
             [roll, pitch, cos(delta_angle_to_target), sin(delta_angle_to_target)],
             0.3 * body_rpy_speed,  # 0.3 is just scaling typical speed into -1..+1, no physical sense here
             0.05 * body_acc,  # 0.05 is just scaling typical speed into -1..+1, no physical sense here
             [z - target_z]]),
            -5, 5)

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
            j.set_servo_target(float(action[n]), 0.045, 0.045, self.JOINT_MAX_TORQUE)

    def _is_done(self, state):
        raise NotImplementedError

    def _calc_reward(self, state, action):
        raise NotImplementedError


class RoboschoolPremaidAIWalkerEnv(RoboschoolPremaidAIEnv):
    PROGRESS_COST_WEIGHT = 10.
    ELECTRICITY_COST_WEIGHT = -2.
    BODY_RP_SPEED_COST_WEIGHT = -0.3
    STALL_TORQUE_COST_WEIGHT = -0.1
    JOINT_POWER_COEF = 0.01
    JOINT_AT_LIMIT_COST_WEIGHT = -0.2
    JOINT_SPEED_EXCEED_LIMIT_COST_WEIGHT = -0.2
    FOOT_SELF_COLLISION_COST_WEIGHT = -1.
    FOOT_SELF_COLLISION_EXCEPTION = {
        'r_foot': {'r_ankle', 'r_lleg'},
        'l_foot': {'l_ankle', 'l_lleg'},
    }

    def __init__(self, action_dim, obs_dim):
        super().__init__(action_dim=action_dim, obs_dim=obs_dim)
        self._last_potential = None
        self._last_joint_speed = None

    def _calc_alive_bonus(self):
        _, _, head_z = self._head.pose().xyz()
        return 2. if 0.3 < head_z < 0.6 else -1.

    def _is_done(self, state):
        return self._calc_alive_bonus() < 0

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

        # calculate alive bonus
        alive_bonus = self._calc_alive_bonus()

        # calculate feet self collision cost
        collisions = [(f.name, part.name) for f in self._feet_objects for part in self.cpp_robot.parts
                      if part.name != f.name
                      and part.name not in self.FOOT_SELF_COLLISION_EXCEPTION[f.name]
                      and np.linalg.norm(np.array(f.pose().xyz()) - np.array(part.pose().xyz())) < 0.06]
        # print(collisions)
        feet_collision_cost = self.FOOT_SELF_COLLISION_COST_WEIGHT * len(collisions)

        # calculate exceed speed limit cost
        joint_speed_exceed_limit_count = np.count_nonzero(np.abs(joint_speed) > self.JOINT_MAX_SPEED)
        joint_speed_limit_cost = self.JOINT_SPEED_EXCEED_LIMIT_COST_WEIGHT * joint_speed_exceed_limit_count

        # calculate body roll, pitch speed
        roll_speed, pitch_speed = state[54], state[55]
        rp_speed_cost = self.BODY_RP_SPEED_COST_WEIGHT * (abs(roll_speed) + abs(pitch_speed))

        self.rewards = [
            progress,
            electricity_cost,
            joints_at_limit_cost,
            joint_speed_limit_cost,
            alive_bonus,
            feet_collision_cost,
            rp_speed_cost,
        ]
        return sum(self.rewards)


class RoboschoolPremaidAIWalker(RoboschoolPremaidAIWalkerEnv):
    def __init__(self):
        # joint position & speed => JOINT_DIM * 2
        # body roll, pitch, cos(delta_angle_to_target), sin(delta_angle_to_target) => 4
        # rpy speed, acc_x, acc_y, acc_z => 6
        # target body height diff => 1
        obs_dim = self.JOINT_DIM * 2 + 4 + 6 + 1
        super().__init__(action_dim=self.JOINT_DIM, obs_dim=obs_dim)


class RoboschoolPremaidAIMimicWalker(RoboschoolPremaidAIWalkerEnv):
    REF_JOINT_ANGLE_REWARD_WEIGHT = 0.65
    REF_JOINT_SPEEd_REWARD_WEIGHT = 0.1

    def __init__(self):
        # joint position & speed => JOINT_DIM * 2
        # body roll, pitch, cos(delta_angle_to_target), sin(delta_angle_to_target) => 4
        # rpy speed, acc_x, acc_y, acc_z => 6
        # target body height diff => 1
        # target phase => 1
        obs_dim = self.JOINT_DIM * 2 + 4 + 6 + 1 + 1
        super().__init__(action_dim=self.JOINT_DIM, obs_dim=obs_dim)
        self._teacher_walk_controller = None
        self._ref_joint_angles = None
        self._ref_joint_angles_absolute = None
        self._ref_joint_angle_speeds = None
        self._terminated = False

    def robot_specific_reset(self):
        super().robot_specific_reset()
        self._teacher_walk_controller = SimpleWalkController(self.scene.dt, 0.91, self.action_space)
        self._terminated = False

    def calc_state(self):
        base_state = super().calc_state()
        ref_joint_angles_absolute, phase = self._teacher_walk_controller.step(self.frame, base_state)

        # convert to relative joint angle
        ref_joint_angle_speeds = ((ref_joint_angles_absolute - self._ref_joint_angles_absolute) / self.scene.dt
                                  if self._ref_joint_angles_absolute is not None else np.zeros(self.JOINT_DIM))
        ref_joint_angles = np.array([np.interp(val, (self.action_space.low[i], self.action_space.high[i]), (-1, 1))
                                     for i, val in enumerate(ref_joint_angles_absolute)])

        self._ref_joint_angles = ref_joint_angles
        self._ref_joint_angle_speeds = ref_joint_angle_speeds / self.JOINT_MAX_SPEED
        self._ref_joint_angles_absolute = ref_joint_angles_absolute
        return np.concatenate([base_state, [phase]])

    def _calc_reward(self, state, action):
        if super()._is_done(state) or self._terminated:
            self._terminated = True
            # set reward to zero for early termination
            self.rewards = [0] * 9
            return 0
        super()._calc_reward(state, action)

        # calculate ref joint angle reward
        joint_angles = state[0::2][:self.JOINT_DIM]
        joint_speed = state[1::2][:self.JOINT_DIM]
        angle_reward = self.REF_JOINT_ANGLE_REWARD_WEIGHT * exp(
            -2 * sum((joint_angles - self._ref_joint_angles)**2))
        angle_speed_reward = self.REF_JOINT_SPEEd_REWARD_WEIGHT * exp(
            -0.1 * sum((joint_speed - self._ref_joint_angle_speeds)**2))
        self.rewards.extend([angle_reward, angle_speed_reward])
        return sum(self.rewards)

    def _is_done(self, state):
        # don't exit episode on early termination
        return False
