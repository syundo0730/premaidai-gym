import os

import numpy as np
from roboschool.scene_abstract import cpp_household
from roboschool.gym_urdf_robot_env import RoboschoolUrdfEnv
from roboschool.multiplayer import SharedMemoryClientEnv
from roboschool.scene_stadium import SinglePlayerStadiumScene


class RoboschoolPremaidAIEnv(SharedMemoryClientEnv, RoboschoolUrdfEnv):
    JOINT_DIM = 25
    OBS_DIM = JOINT_DIM * 2 + 9  # joint position & speed, body roll, pitch, yaw, x, y, z, vx, vy, vz

    def __init__(self):
        model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'models_robot/premaidai_description/urdf/premaidai.urdf')
        RoboschoolUrdfEnv.__init__(self, model_urdf=model_path, robot_name='base_link',
                                   action_dim=self.JOINT_DIM, obs_dim=self.OBS_DIM,
                                   fixed_base=False, self_collision=False)

    def create_single_player_scene(self):
        return SinglePlayerStadiumScene(gravity=9.8, timestep=0.0165 / 8, frame_skip=8)

    def robot_specific_reset(self):
        cpose = cpp_household.Pose()
        cpose.set_xyz(0, 0, 0.25)
        cpose.set_rpy(0, 0, 0)
        self.cpp_robot.set_pose_and_speed(cpose, 0, 0, 0)

        for j in self.ordered_joints:
            j.reset_current_position(0, 0)
        # print([(i, j.name) for i, j in enumerate(self.ordered_joints)])
        self.scene.actor_introduce(self)

    def calc_state(self):
        robot_pose = self.robot_body.pose()
        joint_angles_and_speeds = np.array([j.current_position() for j in self.ordered_joints], dtype=np.float32).flatten()
        return np.concatenate(
            [joint_angles_and_speeds] + [robot_pose.rpy()] + [robot_pose.xyz()] + [self.robot_body.speed()])

    def step(self, action):
        self._apply_action(action)
        self.scene.global_step()
        state = self.calc_state()
        reward = self._calc_reward(state)
        done = self._is_done(state)
        return state, reward, done, {}

    def _apply_action(self, action):
        raise NotImplementedError

    def _is_done(self, state):
        raise NotImplementedError

    def _calc_reward(self, state):
        raise NotImplementedError


class RoboschoolPremaidAIWalker(RoboschoolPremaidAIEnv):
    def _apply_action(self, action):
        for n, j in enumerate(self.ordered_joints):
            j.set_servo_target(action[n], 0.1, 0.1, 40)

    def _is_done(self, state):
        return False

    def _calc_reward(self, state):
        return 1
