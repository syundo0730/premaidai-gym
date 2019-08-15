import time
from collections import deque
from math import radians, atan2, sin, pi

import gym
import numpy as np
import premaidai_gym


class _PhysicalPhaseEstimator:
    def __init__(self):
        self._body_vel_y_queue = deque(maxlen=3)
        self._last_body_vel_y = None
        self._last_body_acc_y = None

    def _filter_body_vel_y(self, body_vel_y):
        self._body_vel_y_queue.append(body_vel_y)
        return np.mean(self._body_vel_y_queue)

    def update(self, body_vel_y, dt):
        body_vel_y = self._filter_body_vel_y(body_vel_y)
        body_acc_y = (body_vel_y - self._last_body_vel_y) / dt if dt else 0
        body_jerk_y = (body_acc_y - self._last_body_acc_y) / dt if dt else 0
        self._last_body_vel_y = body_vel_y
        self._last_body_acc_y = body_acc_y
        return atan2(body_jerk_y, body_acc_y)
        # return atan2(body_acc_y, body_vel_y)


class _Oscillator:
    OMEGA = 6
    K = 6 * 3

    def __init__(self):
        self._angle = 0

    def update(self, ref_angle, dt):
        d_angle = self.OMEGA + self.K * sin(ref_angle - self._angle)
        self._angle += d_angle * dt
        return self._angle


class _CPGWalker:
    LR_DELAY = pi
    PITCH_DELAY = 0.5 * pi
    # AMP_HIP = 0.03
    AMP_HIP = 0.03
    # AMP_ANKLE = 0.03
    AMP_ANKLE = 0.03
    AMP_PITCH = 0.08
    # AMP_PITCH = 0.16
    AMP_h_s = 0.2
    AMP_a_s = 0.2

    def __init__(self, env: gym.Env):
        self._env = env
        self._home_pose = np.full(env.action_space.shape[0], 0.)
        self._home_pose[13] = radians(60)  # right arm
        self._home_pose[18] = radians(-60)  # left arm
        self._last_time = None
        self._phase_estimator = _PhysicalPhaseEstimator()
        self._oscillator = _Oscillator()

    def _update_oscillator(self, body_vel_y, current_time):
        dt = current_time - self._last_time if self._last_time else 0
        self._last_time = current_time
        theta = self._phase_estimator.update(body_vel_y, dt)
        # theta = 2.2 * pi * current_time
        # return self._oscillator.update(theta, dt)
        return 2.2 * pi * current_time

    def action(self, obs):
        body_vel_y = obs[-2]
        l_phi_c = self._update_oscillator(body_vel_y, time.time())
        r_phi_c = l_phi_c + self.LR_DELAY
        l_phi_c_p = l_phi_c + self.PITCH_DELAY
        r_phi_c_p = l_phi_c_p + self.LR_DELAY

        sin_r_phi_c = sin(r_phi_c)
        sin_r_phi_c_p = sin(r_phi_c_p)
        sin_l_phi_c = sin(l_phi_c)
        sin_l_phi_c_p = sin(l_phi_c_p)
        theta_hip_rest = -pi / 10
        theta_knee_rest = pi / 5
        theta_ankle_rest = -pi / 10

        theta_hip_r = self.AMP_HIP * sin_r_phi_c
        theta_ankle_r = -self.AMP_ANKLE * sin_r_phi_c
        # r_th = -A_p * maxOut(sinf_r_phi_c);
        r_th = -self.AMP_PITCH * sin_r_phi_c
        r_theta_hip_p = r_th + theta_hip_rest
        r_theta_knee_p = -2 * r_th + theta_knee_rest
        r_theta_ankle_p = r_th + theta_ankle_rest

        # l_th = -A_p * maxOut(sinf_l_phi_c);
        l_th = -self.AMP_PITCH * sin_l_phi_c
        l_theta_hip_p = l_th + theta_hip_rest
        l_theta_knee_p = -2 * l_th + theta_knee_rest
        l_theta_ankle_p = l_th + theta_ankle_rest

        # r_hteta_hip_s = -self.AMP_h_s * sin_r_phi_c_p
        # r_theta_ankle_s = self.AMP_a_s * sin_r_phi_c_p
        # l_theta_hip_s = -self.AMP_h_s * sin_l_phi_c_p
        # l_theta_ankle_s = self.AMP_a_s * sin_l_phi_c_p

        action = np.zeros_like(self._home_pose)

        action[1] = theta_hip_r
        action[2] = r_theta_hip_p
        action[3] = r_theta_knee_p
        action[4] = r_theta_ankle_p
        # action[1] = theta_hip_rest
        # action[2] = theta_knee_rest
        # action[3] = theta_ankle_rest
        action[5] = theta_ankle_r

        action[7] = theta_hip_r
        action[8] = l_theta_hip_p
        action[9] = l_theta_knee_p
        action[10] = l_theta_ankle_p
        # action[1] = theta_hip_rest
        # action[2] = theta_knee_rest
        # action[3] = theta_ankle_rest
        action[11] = theta_ankle_r

        action[13] = radians(60)  # right arm
        action[18] = radians(-60)  # left arm

        return action


def main():
    env = gym.make('RoboschoolPremaidAIWalker-v0')
    cpg_walker = _CPGWalker(env)

    obs = env.reset()
    while True:
        action = cpg_walker.action(obs)
        # action = np.full(env.action_space.shape[0], 0.)
        # action[13] = radians(60)  # right arm
        # action[18] = radians(-60)  # left arm
        obs, reward, done, _ = env.step(action)
        if done:
            print('hoggeee', done)
            break
        env.render()
        time.sleep(0.02)


if __name__ == '__main__':
    main()
