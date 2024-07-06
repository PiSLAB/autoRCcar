# autoRCcar_outdoor

<p align="center"> <img src="images\autoRCcar_test_video.gif" width="500" height="300"/> </p>

## System
### Hardware
<p1 align="center"> <img src="images\autorccar_hw.jpg" width="655" height="488"/> </p1>
- 1/8 VRX RH818 (RC Car)
- NVIDIA Jetson Nano
  - ubuntu 20.04 + ROS2 Galactic
- SparkFun GPS-16344 (ublox ZED-F9R) + GNSS Antenna
- ESP32 WROOM Devkit
### Software
<p2 align="center"> <img src="images\autorccar_sw.jpg" width="655" height="353"/> </p2>
### Note
- In the file below, you need to set the path of the `autorcar_navigation/config.yaml`.
  `~/autoRCcar_outdoor/autorccar_navigation/src/main_ekf.cpp`
```C++
// Modify the config.yaml path
std::string config = "/home/{User_Name}/{ROS_Workspace}/src/autoRCcar_outdoor/autorccar_navigation/config.yaml";
```
- If the mounted axis of the IMU is changed, the axis must be converted in the path below.
  `~/autoRCcar_outdoor/autorccar_navigation/src/ekf_ros_wrapper.cpp`
```C++
void EKFWrapper::ImuCallback(const autorccar_interfaces::msg::Imu & msg)
{
  // Transform IMU coordinates
  double imu_time = msg.timestamp.sec + msg.timestamp.nanosec*1e-9;
  Eigen::Vector3d acc = Eigen::Vector3d(msg.linear_acceleration.y, msg.linear_acceleration.x, -msg.linear_acceleration.z);
  Eigen::Vector3d gyro = Eigen::Vector3d(msg.angular_velocity.y, msg.angular_velocity.x, -msg.angular_velocity.z);
  ...
}
```
- If the UART of ESP32 is changed, you must change the port in the file below.
  `~/autoRCcar_outdoor/autorccar_control/src/main.cpp`
```C++
// EPS32 device
fid = pp.uart_init("/dev/ttyUSB0");
```
- It is impossible to change the goal point while driving because a goal point is once subscribed to in the control node. You must re-run the `autorccar_control` to enter a new goal point.
- The vehicle speed is currently limited to slow. You can change it in the `autorccar_control`.
## Depenency
- [ROS2](https://docs.ros.org/en/galactic/index.html) (Galactic)
- [Eigen](https://eigen.tuxfamily.org/)
- [PyQt](https://pypi.org/project/PyQt5/) (GCS)
## Packages
```bash
autoRCcar_outdoor
  ├── autorccar_launch : autoRCcar packages lunch file
  │
  ├── autorccar_interfaces : custom interface messages
  ├── autorccar_ubloxf9r : sensor DAQ (GNSS & IMU)
  ├── autorccar_navigation : GNSS/INS EKF
  ├── autorccar_path_planning : bezier curve
  ├── autorccar_control : pure pursuit
  ├── autorccar_rl_control : car control based on reinforcement learning (New)
  │
  ├── autorccar_esp32 : PWM generator (Arduino IDE)
  │
  ├── autorccar_keyboard : manual control
  └── autorccar_gcs : Command & Monitoring
```
## Build
### RC car
```bash
cd ~/ros2_ws
colcon build --symlink-install
source install/setup.bash
```
If there is a custom messages error, proceed with the `autorccar_interfaces` build first.
```bash
colcon build --symlink-install --packages-select autorccar_interfaces
source install/setup.bash
```
### (Optional) Remote PC
```bash
cd ~/ros2_ws/src/autoRCcar_outdoor/autorccar_gcs
pip3 install -r requirement.txt

cd ~/ros2_ws
colcon build --symlink-install --packages-select autorccar_gcs autorccar_keyboard
source install/setup.bash
```
## Application
### Autonomous Driving
```bash
ros2 run autorccar_launch launch.py
```
If you want to run an individual package,
```bash
ros2 run autorccar_ubloxf9r ubloxf9r
ros2 run autorccar_navigation gnss_ins
ros2 run autorccar_path_planning bezier_curve
ros2 run autorccar_control pure_pursuit
```
### GCS &nbsp;[[README]](https://github.com/PiSLAB/autoRCcar_outdoor/blob/main/autorccar_gcs/README.md)
```bash
ros2 run autorccar_gcs autorccar_gcs
```
### (Optional) Keyboard Control
```bash
ros2 run autorccar_keyboard keyboard_control
```
### (New) Reinfocement Learning
To be updated
```bash
cd ~/ros2_ws/src/autoRCcar_outdoor/autorccar_rl_control/autorccar_rl_control
python rl_control.py
```
## Reference
- Control) https://github.com/AtsushiSakai/PythonRobotics
- Keyboard) https://github.com/ros2/teleop_twist_keyboard

## Collaborator
- [GS Park](https://github.com/gspark87)
- [JH Lee](https://github.com/lee90108)
- [JH Bae](https://github.com/luke7637)
- [HJ Son](https://github.com/dlfksj)
