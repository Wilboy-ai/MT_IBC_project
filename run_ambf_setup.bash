#!/bin/bash

# Start the ROS core
roscore &
echo "Waiting for roscore to initialize..."
sleep 5

# Launch the AMBF simulator
cd ~/ambf/bin/lin-x86_64 && ./ambf_simulator --launch_file ~/surgical_robotics_challenge/launch.yaml -l 0,1,3,4,14,15 -p 1000 -t 1 --override_max_comm_freq 1000 &
echo "Waiting for ambf_simulator to initialize..."
sleep 5


# Launch the state publisher
roslaunch accelnet_challenge_sdu state_publisher.launch &
echo "Waiting for state_publisher to initialize..."
sleep 5

# Launch the CRKT interface
python3 ~/surgical_robotics_challenge/scripts/surgical_robotics_challenge/launch_crtk_interface.py &
echo "Waiting for CRTK interface to initialize..."
sleep 5

# Launch RViz
#rviz -d ~/surgical_robotics_challenge/scripts/surgical_robotics_challenge/MT-IBC-SurgicalRobotics/ambfstatepublisher_setup.rviz &
#echo "Waiting for RViz to initialize..."
#sleep 15



