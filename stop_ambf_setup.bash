echo "Closing all processes..."
pkill -f roscore
pkill -f ambf_simulator
pkill -f state_publisher.launch
pkill -f launch_crtk_interface.py
pkill -f rviz
