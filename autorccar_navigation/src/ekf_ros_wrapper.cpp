#include "ekf_ros_wrapper.h"
#include "ekf.h"

#include "rclcpp/rclcpp.hpp"

#include <eigen3/Eigen/Dense>

EKFWrapper::EKFWrapper(EKF* pEKF):Node("ins_gps"), mpEKF(pEKF) 
{
    // // time initialization (-> moved to ImuCallback)
    // rclcpp::Time now = this->now();
    // mpEKF->SetTimeInit(now.seconds());
    // RCLCPP_INFO(this->get_logger(), "sec %lf nsec %ld", now.seconds(), now.nanoseconds());

    imu_subscriber = this->create_subscription<autorccar_interfaces::msg::Imu>("IMU", 10, std::bind(&EKFWrapper::ImuCallback, this, std::placeholders::_1));
    gps_subscriber = this->create_subscription<autorccar_interfaces::msg::Gnss>("GNSS", 10, std::bind(&EKFWrapper::GpsCallback, this, std::placeholders::_1));
    yaw_subscriber = this->create_subscription<std_msgs::msg::Float32>("setyaw_topic", 10, std::bind(&EKFWrapper::InitYawCallback, this, std::placeholders::_1));
    state_publisher = this->create_publisher<autorccar_interfaces::msg::NavState>("nav_topic",10);
}

void EKFWrapper::ImuCallback(const autorccar_interfaces::msg::Imu & msg)
{
    double imu_time = msg.timestamp.sec + msg.timestamp.nanosec*1e-9;
    Eigen::Vector3d acc = Eigen::Vector3d(msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z);
    Eigen::Vector3d gyro = Eigen::Vector3d(msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z);
    //double imu_time = msg.timestamp.sec + msg.timestamp.nanosec*1e-9;
    //Eigen::Vector3d acc = Eigen::Vector3d(msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z);
    //Eigen::Vector3d gyro = Eigen::Vector3d(msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z);

    // Set EKF start time 
    if (!mpEKF->is_time_set)
    {
        mpEKF->SetTimeInit(imu_time);        
    }

    // Update IMU data
    mpEKF->UpdateInputVector(imu_time, acc, gyro);
    
    if (mpEKF->is_initialized)
    {
        RCLCPP_DEBUG(this->get_logger(), "Prediction");
        mpSol = mpEKF->Predict();
        PublishNavSol();
    }
    else
    {
        RCLCPP_DEBUG(this->get_logger(), "Alignment");
        mpEKF->Alignment();
    }
}

void EKFWrapper::GpsCallback(const autorccar_interfaces::msg::Gnss & msg)
{
    double gps_time = msg.timestamp.sec + msg.timestamp.nanosec*1e-9;
    Eigen::Vector3d gps_pos = Eigen::Vector3d(msg.position_ecef.x, msg.position_ecef.y, msg.position_ecef.z);
    Eigen::Vector3d gps_vel = Eigen::Vector3d(msg.velocity_ecef.x, msg.velocity_ecef.y, msg.velocity_ecef.z);
    
    // Update GNSS data
    mpEKF->UpdateMeasurementVector(gps_time, gps_pos, gps_vel);

    if (mpEKF->is_initialized)
    {
        RCLCPP_DEBUG(this->get_logger(), "Correction");

        mpSol = mpEKF->Correct();
        PublishNavSol();
    }
}

void EKFWrapper::InitYawCallback(const std_msgs::msg::Float32 &msg)
{
    // Set Initial yaw [deg]
    // This function resets all variables and restarts the EKF.
    mpEKF->SetYawInit(msg.data);
}

void EKFWrapper::PublishNavSol()
{
    Eigen::Matrix<double, 20, 1> x = mpSol->GetNavRosMsg();

    auto message = autorccar_interfaces::msg::NavState();
    message.timestamp.sec = (int)x(0);
    message.timestamp.nanosec = (x(0)-(int)x(0))*1e9;
    message.origin.x = x(1);
    message.origin.y = x(2);
    message.origin.z = x(3);
    message.position.x = x(4);
    message.position.y = x(5);
    message.position.z = x(6);
    message.velocity.x = x(7);
    message.velocity.y = x(8);
    message.velocity.z = x(9);
    message.quaternion.w = x(10);
    message.quaternion.x = x(11);
    message.quaternion.y = x(12);
    message.quaternion.z = x(13);
    message.acceleration.x = x(14);
    message.acceleration.y = x(15);
    message.acceleration.z = x(16);
    message.angular_velocity.x = x(17);
    message.angular_velocity.y = x(18);
    message.angular_velocity.z = x(19);

    state_publisher->publish(message);
}
