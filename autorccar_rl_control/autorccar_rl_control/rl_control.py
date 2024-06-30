import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import serial
import struct
import numpy as np
import autoRCcar_gym
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import sys, ast
# from stable_baselines3 import PPO
from sb3_contrib import TQC

from autorccar_interfaces.msg import NavState
from std_msgs.msg import Int8
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Vector3



class RLControl(Node):
    def __init__(self):
        super().__init__('rl_control')

        # Configure QoS profile for publishing and subscribing
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        ## Serial Port
        self.ser = serial.Serial(
            port='/dev/ttyUSB0',  # 포트 이름
            baudrate=115200,        # 보드레이트
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            bytesize=serial.EIGHTBITS,
            timeout=1             # 타임아웃
        )
        if self.ser.is_open:
            print("Serial port is already open.")
        else:
            self.ser.open()

        ## Variable
        self.command = None
        self.key_steer = None
        self.key_esc = None
        self.mode = None
        self.dt = 0.1

        self.ESC_PWM_MIN = 3277
        self.ESC_PWM_N = 4915
        self.ESC_PWM_MAX = 6553
        self.Steer_PWM_MIN = 10  # 0 (margin 10)
        self.Steer_PWM_N = 90
        self.Steer_PWM_MAX = 170  # 180 (margin 10)

        ## Create publishers
        self.pub_ctrl = self.create_publisher(Vector3, 'param_ctl', qos_profile)
        # self.timer = self.create_timer(0.5, self.timer_callback)

        ## Create subscribers
        self.sub_nav = self.create_subscription(NavState, 'nav_topic', self.callback_navigation, qos_profile)
        self.sub_gcs = self.create_subscription(Int8, 'command_topic', self.callback_gcs, qos_profile)
        self.sub_key = self.create_subscription(Twist, 'cmd_vel', self.callback_keyboard, qos_profile)


        ## RL initialize
        self.env = gym.make('avoid-v1')
        self.model = TQC.load('avoid-v1_tqc2', env=self.env)

        wp = []  ## Set waypoint
        with open("waypoints.txt", 'r') as file:
            for line in file:
                line = line.strip()
                try:
                    point = list(map(float, line.split()))
                    rounded_point = [round(num, 1) for num in point]
                    wp.append(rounded_point)
                except:
                    pass
        self.env.wp_end = len(wp)

        wp_idx = 0  ## Rest env
        self.obs, info = self.env.reset(set_goal=wp[wp_idx])
        print("\n\nWaypoint Index = ", wp_idx)
        print(info['task'])
        print("init_state : ", info['state'], "heading[deg] : ", np.degrees(info['state'][2]))
        print("goal : ", info['goal'])
        print("obstacle : ", info['obstacle'])
        print("---------------------")
        print("obs : ", self.obs)
        wp_idx += 1


    def callback_navigation(self, msg):
        """Callback function for navigation topic subscriber."""
        self.obs = self.calculate_observation

        action, _states = self.model.predict(self.obs, deterministic=True)
        # self.obs, reward, terminated, _, info = self.env.step(action)

        self.uart_tx(action)
        self.publish_control_input(action)


    def callback_gcs(self, msg):
        """Callback function for GCS topic subscriber."""
        self.command = msg.data

    def callback_keyboard(self, msg):
        """Callback function for keyboard control topic subscriber."""
        self.key_steer = msg.angular.z
        self.key_esc = msg.linear.x

    def calculate_observation(self):
        ## obs :  [8.8  8.3  0.98916256  4.7  4.35  0.9985859  3.9  1.7453293  0.  ]
        self.obs[0] = 1
        return self.obs


    def uart_tx(self, action):
        sf = -1.5
        fa = 206.0306
        fb = -40.6541
        fc = 5138.7189

        rl_steering = action[0]
        rl_accel = action[1]


        if (self.dt > 1):
            self.dt = 0.1

        # delta = 45 *(M_PI/180);
        conStr = int(sf* rl_steering * (180/np.pi)) + self.Steer_PWM_N # [deg]

        # newVel = x_vel + ai * dt;  // [m/s]
        newVel = rl_accel;  # [m/s]
        conVel = int((fa*newVel) + (fb*newVel*newVel) + fc)

        if (self.command == 0):  # motor off
            conVel = self.ESC_PWM_N

        if (self.command == 2):
            conStr = self.Steer_PWM_N + self.key_steer
            conVel = self.ESC_PWM_N + self.key_esc

        if (conStr >= self.Steer_PWM_MAX):
            conStr = self.Steer_PWM_MAX
        if (conStr <= self.Steer_PWM_MIN):
            conStr = self.Steer_PWM_MIN

        if (conVel >= 5350):  # ESC_PWM_MAX
            conVel = 5350
        if (conVel <= self.ESC_PWM_MIN):
            conVel = self.ESC_PWM_MIN

        msgs = struct.pack('>BBHHHH', 0xff, 0xfe, conStr, conVel, self.command >> 8, self.command & 0xff)
        self.serial_port.write(msgs)


    def publish_control_input(self, act):
        msg = Vector3()
        msg.x = act[0]  # steering
        msg.y = act[1]  # accel
        msg.z = self.mode
        self.pub_ctrl.publish(msg)




def main(args=None):
    print("autoRCcar Control based on Reinforcement Learning")
    rclpy.init(args=args)
    rl_control = RLControl()
    rclpy.spin(rl_control)
    rl_control.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
