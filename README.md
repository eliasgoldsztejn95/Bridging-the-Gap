# Bridging-the-Gap
Regularized Reinforcement Learning for Improved Classical Motion Planning with Safety Modules


https://github.com/eliasgoldsztejn95/Bridging-the-Gap/assets/75029654/dd3b88de-16da-475e-9408-4f19cf3a2298

# Requirements

 - ROS 1
 - Python version at least 3.6
 - Python packages: pytorch, rospy

## Running on a real robot


 - Download real_robot folder to the robot computer
 - Use move_base package for generating global plan and goal
 - Run cdrl.py


## Training on simulation
- Create your simulation environment of choice. In our case we used a realistic hospital world taken from: https://github.com/aws-robotics/aws-robomaker-hospital-world. We integrated moving people using: https://github.com/srl-freiburg/pedsim_ros. The steps for combining them can be found in **PTDRL Repository - Simulation**
- Download your robot. In our case we used Turtlebot with waffle-pi.
- Clone folder. 
- Train RL: train.py.  **save_model=True** arg to save neural networks.
- Train Supervisor: train_supervisor.py.

## Simulation, robot and training algorithm
- **Bridging the Gap** was written in the OpenAI ROS fashion. This means that training, simulation, and robot environments are separated.
- task_env.py provides all the context we want the robot to learn for the RL task, in this case, navigating fast and safely. It contains the main functions: **step** **reset** and **_init_env_variables**. It includes the move_base related functions: **_set_init_pose**, **_send_goal**, and **tune_parameters**.
- robot_en.py contains all the functions associated to the specific robot that you want to train. It also provides the connection with move_base. It contains the main callback functions: **_scan_callback**, **_odom_callback**, and **_costmap_callback**. 
- params.yaml includes:
1. Which local planner is used
2. The set of parameters of the local planner
3. A list of rooms the robot goes through inside the environment

# Hospital world
<img src="https://user-images.githubusercontent.com/75029654/166143327-e4caf24c-6b8a-4629-9f03-982de54fe37e.png" width="300" height="300">

The simulator of the hospital was taken from: https://github.com/aws-robotics/aws-robomaker-hospital-world.
which is amazon's representation of a hospital. It is very rich in the sense of quantity and quality of objects simulated, and it represents 
realistically a hospital.

### Notes
The models have to be downloaded manually from: https://app.ignitionrobotics.org/fuel/models into local models folder.

# Pedestrain simulator
<img src="https://user-images.githubusercontent.com/75029654/166143081-f978b80b-680e-4c15-87a3-a95c89352896.png" width="500" height="250">

The pedestrian simulation is acquired using pedsim_ros package. The package is based on: https://arxiv.org/pdf/cond-mat/9805244.pdf social force model.
This package allows to choose the quantity and type of pedestrians, and motion patterns. THe package was taken from: https://github.com/srl-freiburg/pedsim_ros.

### Notes
The hospital world of amazon has to be moved to pedsim_gazebo_plugin/worlds folder. Notice that the line: plugin name="ActorPosesPlugin" filename="libActorPosesPlugin.so"
has to be added at the end of the file to allow pedestrian movement.
  
Notice that the pedestrian simulator has to account for obstacles in the world. This should be described in <scenario>.xml found in pedsim_simulator/secnarios.
