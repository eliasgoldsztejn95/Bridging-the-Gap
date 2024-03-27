#!/usr/bin/env python3

#######################
## Environment class ##
#######################

# Reset environment
# Compute reward
# Perform action
# Get obs
# Is done


import rospy
import actionlib
from gazebo_msgs.srv import SetModelState, SetModelStateRequest
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from geometry_msgs.msg import Pose, PoseWithCovarianceStamped , PoseStamped, Twist
from std_srvs.srv import Empty, EmptyResponse, EmptyRequest
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import numpy as np
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal, MoveBaseFeedback, MoveBaseResult
from actionlib_msgs.msg import GoalStatusArray
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from actionlib_msgs.msg import GoalID
import csv
import os
from pathlib import Path
from std_srvs.srv import EmptyRequest,  Empty
from visualization_msgs.msg import MarkerArray, Marker
import dynamic_reconfigure.client
import robot_env
import lstm_mdn_simple
import yaml
from yaml.loader import SafeLoader
import torch
import phydnet_predict
from models.vae import VAE
from models.mdrnn import MDRNNCell
from models.reward import REWARD
from models.sft import SFT
from models.simple_vit import SimpleViT
import time
import random

import matplotlib.cm as cm
from random import randrange
import matplotlib.animation as animation
from PIL import Image
import time
from matplotlib.animation import FuncAnimation
from numpy import inf
from scipy.stats import truncnorm
import copy
import skfuzzy as fuzz


import matplotlib.pyplot as plt

from scipy.ndimage import rotate

# open yaml file
dir_path = os.path.dirname(os.path.realpath(__file__))

# Paths
#dir_path_phydnet = dir_path + "/networks/phydnet.pth"
dir_path_phydnet = dir_path + "/networks/encoder_phydnet_0_10.pth"
dir_path_vae_input = dir_path + "/networks/vae_input_2/best.tar"
dir_path_vae_input_inflation = dir_path + "/networks/vae_input_inflation/best.tar"
dir_path_vae_prediction = dir_path + "/networks/vae_prediction_2/best.tar"
dir_path_mdrnn = dir_path + "/networks/mdrnn/best.tar"
dir_path_reward = dir_path + "/networks/reward/checkpoint_supervised_reward1_96_26.pth"
dir_path_sft = dir_path + "/networks/sft/checkpoint_supervised_dqn88_30.pth"
dir_path_cdrl_human_reward = dir_path + "/networks/cdrl/simple_vit_reward_random.pth"

dir_path_yaml = dir_path
dir_path_yaml = dir_path_yaml.split("/")
dir_path_yaml = dir_path_yaml[:-2]
dir_path_yaml += ["params"]
dir_path_yaml = '/'.join(dir_path_yaml)

file = "task_params_dpo2"
#file = "task_params_supervisor"
yaml_file = "/" + file + ".yaml"

dir_path_yaml += yaml_file

# Constants
ASIZE, LSIZE, RSIZE, RED_SIZE, SIZE =\
    3, 32, 256, 64, 64

class PtdrlTaskEnv(robot_env.PtdrlRobEnv):

    """ Superclass for PTDRL Task environment
    """
    def __init__(self):
        """Initializes a new PTDRL Task environment
        """

        self.get_params()

        # Simulation rate
        self.rate_task = rospy.Rate(10) # Use 10 rate. 5 now 3

        # Counters
        self.counter = -1 #1 Counter to change goal and init pose
        self.timeout_action = 1 # After how many frames to do each action. 3 is around one second. Before 3 # 20 or 10 (2 sec or 1). 6 now 20
        self.num_obs_for_prediction = 3 # How many frames needed for prediction
        self.clear_costmap_counter = 0  
        self.stuck = 0 # if more than 500 seconds passed and the robot did not move then the environment crashed      
        self.wp_counter = 0
        # Position and goals
        self.init_pose = None
        self.goal = None
        self.wp_goal = np.zeros(2)
        #self.offset  = [16.4, 11.9]
        self.offset = [0, 0]
        # Robot constants
        self.max_vel_robot = 1
        # Random action
        self.random_counter = 0 # Counter for random actions
        self.random_noise_vel = 0 # Random noise for action
        self.random_noise_ang= 0
        # Supervisor
        self.supervisor_counter = 0
        self.max_supervisor = 8 # When activated wait one second

        self.total_distance_steps = 0
        self.total_distance_no_steps = 0

        self.counter_gp = 0

        # NN's
        self.img_channels = 3
        self.latent_space = 10
        self.cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda else "cpu")
        #self.prediction_module = lstm_mdn_simple.LSTM_MDN(dir_path_checkpoint)

        # Initialize networks
        #self.phydnet = phydnet_predict.PhydNet(dir_path_phydnet)
        #self.vae_input = VAE(self.img_channels, self.latent_space).to(self.device)
        self.vae_input_inflation = VAE(self.img_channels, 32).to(self.device)
        #self.vae_prediction = VAE(self.img_channels, self.latent_space).to(self.device)
        self.mdrnn = MDRNNCell(LSIZE, ASIZE, RSIZE, 5).to(self.device)

        # SFT and Reward
        self.reward_net = REWARD(290, 4).to(self.device)
        self.sft = SFT(290, 4).to(self.device)

        # CDRL reward
        self.reward_cdrl = SimpleViT(
                            image_size=(60, 60),
                            patch_size=(6, 6), 
                            dim=128, 
                            depth=4,
                            heads=3, 
                            mlp_dim=256,
                            output_dim = 1
                            ).to(self.device)
        
        # Load networks
        #self.state_input = torch.load(dir_path_vae_input)
        #self.vae_input.load_state_dict(self.state_input['state_dict'])

        self.state_input_inflation = torch.load(dir_path_vae_input_inflation)
        self.vae_input_inflation.load_state_dict(self.state_input_inflation['state_dict'])

        # SFT and Reward
        self.reward_net.load_state_dict(torch.load(dir_path_reward))
        self.sft.load_state_dict(torch.load(dir_path_sft))

        # CDRL reward
        self.reward_cdrl.load_state_dict(torch.load(dir_path_cdrl_human_reward))

        #state_prediction = torch.load(dir_path_vae_prediction)
        #self.vae_prediction.load_state_dict(state_prediction['state_dict'])

        self.state_mdrnn = torch.load(dir_path_mdrnn)
        self.mdrnn.load_state_dict({k.strip('_l0'): v for k, v in self.state_mdrnn['state_dict'].items()})

        # INitialize netowrk variables
        self.hidden =[torch.zeros(1, RSIZE).to(self.device) for _ in range(2)]

        robot_env.PtdrlRobEnv.__init__(self, model_name = self.model_name, amcl = self.amcl, min_dist_to_obstacle = self.min_dist_to_obstacle,
                                    min_dist_to_goal = self.min_dist_to_goal, num_tracks = self.num_tracks, timeout = self.timeout)

    def get_params(self):

        with open(dir_path_yaml, 'r') as f:
            data = list(yaml.load_all(f, Loader=SafeLoader))
        
        # Time
        self.timeout = data[0]["timeout"]

        # Model
        self.model_name = data[0]["model_name"]

        # Navigation
        self.amcl = data[0]["amcl"]

        # Tracking
        self.num_tracks = data[0]["num_tracks"]

        # Actionsctions
        self.discrete_actions = data[0]["discrete_actions"]
        self.list_tune_params = []

        # World
        self.min_dist_to_obstacle = data[0]["min_dist_to_obstacle"]
        self.min_dist_to_goal = data[0]["min_dist_to_goal"]

        self.list_init_pose = []
        self.list_goals = []
        self.sub_goal = 0

        # Local planner
        self.local_planner = data[0]["local_planner"]
        if self.local_planner == "dwa":
            tune_params = "tune_params_dwa"
        else:
            tune_params = "tune_params_teb"

        for i in range(len(data[0][tune_params])):
            one_param = []
            one_param.append(data[0][tune_params][i]["name"])
            one_param.append([data[0][tune_params][i]["min"], data[0][tune_params][i]["max"]])
            self.list_tune_params.append(one_param)
        #print(f"These are the params {self.list_tune_params}")

        # Simple init goal
        # for i in range(len(data[0]["list_init_pose"])):
        #     self.list_init_pose.append(Pose())
        #     self.list_goals.append(Pose())
            
        #     #print(data[0]["list_init_pose"][0]["x"])
        #     self.list_init_pose[i].position.x = data[0]["list_init_pose"][i]["x"]
        #     self.list_init_pose[i].position.y = data[0]["list_init_pose"][i]["y"]
        #     self.list_init_pose[i].position.z = 0
        #     orientation = quaternion_from_euler(0,0,0)
            
        #     self.list_init_pose[i].orientation.x =  orientation[0]
        #     self.list_init_pose[i].orientation.y =  orientation[1]
        #     self.list_init_pose[i].orientation.z =  orientation[2]
        #     self.list_init_pose[i].orientation.w =  orientation[3]

        #     self.list_goals[i].position.x = data[0]["list_goals"][i]["x"]
        #     self.list_goals[i].position.y = data[0]["list_goals"][i]["y"]

        # # # Curriculum init goal
        for i in range(len(data[0]["list_init_pose"])): 
            self.list_init_pose.append(Pose())
            self.list_goals.append([])

            self.list_init_pose[i].position.x = data[0]["list_init_pose"][i]["x"]
            self.list_init_pose[i].position.y = data[0]["list_init_pose"][i]["y"]

            for j in range(len(data[0]["list_goals"][i])):
                self.list_goals[i].append(Pose())
                value = data[0]["list_goals"][i][j]["x"]
                self.list_goals[i][j].position.x = data[0]["list_goals"][i][j]["x"]
                self.list_goals[i][j].position.y = data[0]["list_goals"][i][j]["y"]
    
    def _set_init_pose(self):

        self.init_robot(self.init_pose)
        #print(self.init_pose)
        #print("Robot initialized!!!")
    
    def _send_goal(self):

        #print(self.goal)
        self.send_goal(self.goal)
        #print("Goal sent!!!")
    
    def reset(self):
        # Init variables, set robot in position, clear costmap and send goal.

        self.counter += 1 # self.counter += 1 7 14
        iter = self.counter % len(self.list_init_pose)
        print(f"len(self.list_init_pose): {len(self.list_init_pose)}")

        #self.init_pose = self.list_init_pose[iter][self.sub_goal]
        self.goal = self.list_goals[iter][self.sub_goal]

        self.init_pose = self.list_init_pose[iter]
        #self.goal = self.list_goals[iter]

        self._init_env_variables()

        self._set_init_pose()

        self.clear_costmap()

        self._send_goal()

        self.set_wp_goal() # ADDED

        #return np.zeros([290])
        #return np.zeros([60,60]) # Lidar: 722, VAE: 34, VAE_MDRNN: 290, cdrl: 60x60
        ##return np.zeros(2), np.zeros(2), np.zeros([60,60]), 0
        return np.zeros(2), np.zeros(2), np.zeros([120,120]), 0
        #return np.zeros(2), np.zeros(2), np.zeros([2, 120,120]), 0
        #return np.zeros(2), np.zeros([2,60,60]), 0
        #return np.zeros(2), np.zeros(2), np.zeros([2,60,60]), 0
    
    def _init_env_variables(self):

        self.status_move_base = 0
    
    def switch_goal(self, subgoal):

        self.sub_goal = subgoal

    
    def step_old(self, action):
        # Set action,  wait for rewards, obs and is_done
        # This function waits for around 1 second to return results.

        # Reward explanation
        # We want to give frequent rewards to encourage:
        # 1) Speed
        # 2) Collision avoidance
        # The way we achieve this is to give a negative reward for each time step, 
        # and a positive reward for distance covered after each time step.
        self.clear_min_scan_buffer_size()
        self.clear_odom_buffer_size()
        #print(self.get_min_scan_buffer_size())
        #self.clear_costmap()
        self.animate_distance_to_obstacle()
        distance_covered = 0
        robot_position = self.get_odom()

        rewards = []
        obs = []
        is_dones = []
        #print("start counting !!!")
        action = action
        if self.discrete_actions == False:
            action = self.translate_continous_action(action)
        
        self.tune_parameters(action)

        total_distance_covered = 0
        start_pos = 0
        final_pos = 0

        for i in range(self.timeout_action):
            self.rate_task.sleep()
            #print(i)
            bad_update = False

            robot_new_position = self.get_odom()
            distance_covered = self.get_covered_distance(robot_position, robot_new_position)
            total_distance_covered += distance_covered
            #print(f"Distance covered is {distance_covered}")
            robot_position = robot_new_position

            is_done = 0
            reward = 0

            ########## To account for bad updates #########
            if distance_covered == 0:
                bad_update = True

            if not bad_update:  ### Distance reward. Reward longer distances.
                reward = self.adjust_covered_distance(distance_covered) # Before -0.01 -0.005. Now it is between 0 and -1
            reward = -0.07
            #print(f"Distance covered: {distance_covered}")
            #print(f"Adjusted distance covered is {reward}")
            hitted_obstacle, dist_to_obstacle = self.hit_obstacle_2()
            if hitted_obstacle or self.status_move_base == 4: # 4 stands for aborted
                #is_done = 1
                #reward = self.reward_dist_to_goal() # Before -1
                if hitted_obstacle:
                    #reward = self.hit_obstacle_reward(dist_to_obstacle) # -30, -40

                    if not bad_update: ### Hitting reward. Punish shorter hits. Between 0 to -3
                        reward += self.adjust_hit_rewards(dist_to_obstacle)
                        #print(f"Hitted reward: {self.adjust_hit_rewards(dist_to_obstacle)}")
                else:
                    reward = -20
                pass
            if self.status_move_base == 4:
                is_done = 1
                reward = -40
            if self.status_move_base == 3: # 3 stands for goal reached
                is_done = 1
                reward = 0
            rewards.append(reward)

            is_dones.append(is_done)

        #print(f"{total_distance_covered} --  {self.get_covered_distance(start_pos, final_pos)}")
            
        obs = self.get_costmap_buffer()

        reward = self.proccess_rewards_3(rewards)
        #if total_distance_covered == 0: # If Gazebo is stuck the robot does not move
            #reward = 0

        mu_i_o, logvar_i_o, mu_p_o, logvar_p_o = self.process_obs_2(obs)
        is_done = self.proccess_is_dones(is_dones)

        obs = self.extract_2(mu_i_o, logvar_i_o, mu_p_o, logvar_p_o)

        #print("stop counting!!!")
        return obs, reward, is_done

    def step_vae(self, action):
        # Step for vae
        # Set action,  wait for rewards, obs and is_done
        self.clear_min_scan_buffer_size()
        self.clear_odom_buffer_size()
        # Clear costmap every 0.5 seconds
        self.clear_costmap_counter += 1
        if self.clear_costmap_counter == 10:
            self.clear_costmap()
            self.clear_costmap_counter = 0

        # Initialization
        is_done = 0
        action = action
        if self.discrete_actions == False:
            action = self.translate_continous_action(action)
        
        # Take action
        self.tune_parameters(action)
        self.rate_task.sleep()

        # Get observations
        lidar = self.get_filtered_scan()
        obs_costmap = self.get_costmap()
        mu_i_o, logvar_i_o = self.process_obs_3(obs_costmap)
        vel = self.get_vel()
        obs = self.extract_3(mu_i_o, logvar_i_o, vel) # Mistake, do not return logvar

        # Set reward
        reward = self.reward_func(lidar)

        # Episode finished
        if self.status_move_base == 4: # 4 stands for goal aborted
            is_done = 1
            reward = -10
        if self.status_move_base == 3: # 3 stands for goal reached
            is_done = 1
            reward = 0

        return obs, reward, is_done

    def step_vae_mdrnn(self, action):
        # Step for vae and mdrnn
        # Update hidden given action, set action,  wait for rewards, return obs and is_done

        # Ablation
        ablation = 2 # 0 Just lidar, 1 just VAE

        # clear buffers
        self.clear_min_scan_buffer_size()
        self.clear_odom_buffer_size()
        # Clear costmap every 0.5 seconds
        self.clear_costmap_counter += 1
        if self.clear_costmap_counter == 10:
            self.clear_costmap()
            self.clear_costmap_counter = 0

        # Initialization
        is_done = 0
        action = action
        if self.discrete_actions == False:
            action = self.translate_continous_action(action)
        
        # Update hidden
        obs_costmap = self.get_costmap()
        latent_mu, logvar_i_o = self.process_obs_3(obs_costmap)
        mdrnn_action = self.get_action_for_mdrnn(action)
        with torch.no_grad():
            _, _, _, _, _, self.hidden = self.mdrnn(mdrnn_action, latent_mu, self.hidden)


        # Take action
        self.tune_parameters(action)
        self.rate_task.sleep()

        # Get observations
        lidar = self.get_filtered_scan()
        obs_costmap = self.get_costmap()
        latent_mu, _ = self.process_obs_3(obs_costmap)
        vel = self.get_vel()
        obs = self.extract_3(latent_mu, self.hidden[0], vel)
        obs_0 = self.extract_lidar(lidar, vel)
        obs_1 = self.extract_vae(latent_mu, vel)

        angle = self.get_local_goal_angle(1.75)
        print(f"Angle is: {angle[0]}")

        # Set reward
        reward = self.reward_func(lidar)

        # Episode finished
        if self.status_move_base == 4: # 4 stands for goal aborted
            is_done = 1
            reward = -10
        if self.status_move_base == 3: # 3 stands for goal reached
            is_done = 1
            reward = 0

        if ablation == 0:
            return obs_0, reward, is_done
        elif ablation == 1:
            return obs_1, reward, is_done
        else:
            return obs, reward, is_done

    def step_vae_mdrnn(self, action):
        # Step for vae and mdrnn
        # Update hidden given action, set action,  wait for rewards, return obs and is_done

        # Ablation
        ablation = 2 # 0 Just lidar, 1 just VAE

        # clear buffers
        self.clear_min_scan_buffer_size()
        self.clear_odom_buffer_size()
        # Clear costmap every 0.5 seconds
        self.clear_costmap_counter += 1
        if self.clear_costmap_counter == 10:
            self.clear_costmap()
            self.clear_costmap_counter = 0

        # Initialization
        is_done = 0
        action = action
        if self.discrete_actions == False:
            action = self.translate_continous_action(action)
        
        # Update hidden
        obs_costmap = self.get_costmap()
        latent_mu, logvar_i_o = self.process_obs_3(obs_costmap)
        mdrnn_action = self.get_action_for_mdrnn(action)
        with torch.no_grad():
            _, _, _, _, _, self.hidden = self.mdrnn(mdrnn_action, latent_mu, self.hidden)


        # Take action
        self.tune_parameters(action)
        self.rate_task.sleep()

        # Get observations
        lidar = self.get_filtered_scan()
        obs_costmap = self.get_costmap()
        latent_mu, _ = self.process_obs_3(obs_costmap)
        vel = self.get_vel()
        obs = self.extract_3(latent_mu, self.hidden[0], vel)
        obs_0 = self.extract_lidar(lidar, vel)
        obs_1 = self.extract_vae(latent_mu, vel)

        # Set reward
        reward = self.reward_func(lidar)

        # Episode finished
        if self.status_move_base == 4: # 4 stands for goal aborted
            is_done = 1
            reward = -10
        if self.status_move_base == 3: # 3 stands for goal reached
            is_done = 1
            reward = 0

        if ablation == 0:
            return obs_0, reward, is_done
        elif ablation == 1:
            return obs_1, reward, is_done
        else:
            return obs, reward, is_done

    def step_cdrl(self, action, train, with_random, control):
        # Step for vae and mdrnn
        # Update hidden given action, set action,  wait for rewards, return obs and is_done

        # Ablation
        ablation = 2 # 0 Just lidar, 1 just VAE

        # clear buffers
        self.clear_min_scan_buffer_size()
        self.clear_odom_buffer_size()
        # Clear costmap every 0.5 seconds
        self.clear_costmap_counter += 1
        if self.clear_costmap_counter == 10:
            self.clear_costmap()
            self.clear_costmap_counter = 0

        # Initialization
        is_done = 0
        #action = action
        if self.discrete_actions == False:
            action = self.translate_continous_action(action)
        
        # Update hidden
        obs_costmap = self.get_costmap()
        costmap_np = self.costmap_to_np(obs_costmap)
        los = 1.75
        angle_atan, PSI, gp, gp_len = self.get_local_goal_angle(los)
        # If no path is found terminate
        if isinstance(gp, bool) and gp is False:
            is_done = 1
            gp = np.zeros([1,2])

        angle = np.zeros([2])
        angle[0] = np.sin(angle_atan)
        angle[1] = np.cos(angle_atan)
        #print(f"angle: {angle}")
        costmap_np, costmap_np_angle = self.add_angle_to_costmap(costmap_np, angle, PSI)
        #print(f"Minimum: {np.min(costmap_np)}, Maximum: {np.max(costmap_np)}")
        latent_mu, logvar_i_o = self.process_obs_3(obs_costmap)
        if train:
            mdrnn_action = self.get_action_for_mdrnn(action)
        else:
            inflation = {'inflation_radius': 0.35}
            mdrnn_action = self.get_action_for_mdrnn(inflation)
        with torch.no_grad():
            _, _, _, _, _, self.hidden = self.mdrnn(mdrnn_action, latent_mu, self.hidden)


        # Take action
        if train:
            dwa_action = self.get_cmd_vel_dwa()

            if with_random:
                action = self.randomize_action(dwa_action)
                self.send_vel(action)

                costmap_np_angle_and_action = self.add_action_to_costmap(costmap_np_angle, action)
                comand_vel = action
            else:
                self.tune_parameters(action)
                self.send_vel(dwa_action)
                print(dwa_action)
                costmap_np_angle_and_action = self.add_action_to_costmap(costmap_np_angle, dwa_action)
                comand_vel = dwa_action
                odom = self.get_odom()
                #print(odom.twist)
        else:
            #print(f"send_vel(action): {action}")
            # In case robot is not safe
            if self.supervisor():
                action = self.get_cmd_vel_dwa()
            self.send_vel(action)
            

        self.rate_task.sleep()

        # Get observations
        lidar = self.get_filtered_scan()
        obs_costmap = self.get_costmap()
        latent_mu_next, _ = self.process_obs_3(obs_costmap)
        vel = self.get_vel()
        #comand_vel = self.get_comand_vel()
        obs = self.extract_3(latent_mu_next, self.hidden[0], vel)
        obs_0 = self.extract_lidar(lidar, vel)
        obs_1 = self.extract_vae(latent_mu_next, vel)

        #print(f"Angle is: {angle[0]}")

        # Set reward
        reward = self.reward_func(lidar)

        # Episode finished
        if self.status_move_base == 4: # 4 stands for goal aborted
            is_done = 1
            reward = -10
        if self.status_move_base == 3: # 3 stands for goal reached
            is_done = 1
            reward = 0
        if ablation == 0:
            return obs_0, reward, is_done
        elif ablation == 1:
            return obs_1, reward, is_done
        else:
            return obs, reward, is_done, angle, PSI, gp, gp_len, vel, comand_vel, latent_mu.cpu().data.numpy()[0], costmap_np, costmap_np_angle_and_action


    def step_cdrl_short(self, action):
        # Update hidden given action, set action,  wait for rewards, return obs and is_done
        # clear buffers
        self.clear_min_scan_buffer_size()
        self.clear_odom_buffer_size()
        # Clear costmap every 0.5 seconds
        self.clear_costmap_counter += 1
        if self.clear_costmap_counter == 10: # Before 10
            self.clear_costmap()
            self.clear_costmap_counter = 0

        # Initialization
        is_done = 0

        velocity = np.zeros(2)
        #observation = np.zeros([2,60,60])
        observation = np.zeros([2,120,120])
        
        # Get state
        obs_costmap = self.get_costmap()
        costmap_np = self.costmap_to_np(obs_costmap)
        max_tries = 5
        los = 1.75
        tries, angle_atan, PSI, gp, gp_len, wp = self.try_get_path(max_tries, los) # Get waypoint angle
        if tries == max_tries: # If path is not found
            return np.zeros(2), np.zeros(2), costmap_np, -1, 1 #-100
        
        angle = np.zeros([2])
        angle[0] = np.sin(angle_atan)
        angle[1] = np.cos(angle_atan)
        rotated_costmap, costmap_np_angle = self.add_angle_to_costmap(costmap_np, angle, PSI)
        if (np.max(rotated_costmap) - np.min(rotated_costmap)) != 0.0:
            rotated_costmap = (rotated_costmap - np.min(rotated_costmap)) / (np.max(rotated_costmap) - np.min(rotated_costmap))
        observation[0] = rotated_costmap

        dwa_action = self.get_cmd_vel_dwa()

        init_point, wp = self.get_dist_and_wp()
        if isinstance(wp, bool):
            return np.zeros(2), np.zeros(2), costmap_np, -1, 1  

        # Take action
        activate_supervisor = False
        if activate_supervisor:
            self.supervisor_counter = self.supervisor_short_v2()
            if self.supervisor_counter != 0:
                print("Supervisor")
                print(f"dwa: {dwa_action}")
                print(f"action: {action}")
                print(f"supervisor_counter: {self.supervisor_counter}")
                if self.supervisor_counter == 1:
                    self.send_vel(np.asarray([0,0]))
                self.send_vel(dwa_action)
            else:
                # Proto supervisor
                self.send_vel(action)
        else:
            #supervisor_general, positive = self.mini_supervisor()
            supervisor_general = False
            positive = False
            if supervisor_general:
                if positive:
                    self.send_vel(np.asarray([-0.3, 0]))
                else:
                    self.send_vel(np.asarray([0.3, 0]))
            else:
                self.send_vel(action)
            #self.send_vel(action)
        for i in range(5): # Before 5
            self.rate_task.sleep()
            #self.send_vel(action)

        # Get reward
        #reward  = self.reward_function_cdrl(costmap_np_angle, action, dwa_action)
        traveled_dist = self.get_dist_traveled(init_point, wp)
        odom = self.get_odom()
        linear_x = odom.twist.twist.linear.x
        angular_z = odom.twist.twist.angular.z
        reward = self.reward_function_cdrl_short(traveled_dist, linear_x)
        is_done = self.goal_reached_dist()
   

        # Get next state
        X, Y, Z, PSI, qt = self.get_robot_status()
        velocity[0] = linear_x
        velocity[1] = angular_z
        obs_costmap = self.get_costmap()
        costmap_np = self.costmap_to_np(obs_costmap)
        tries, angle_atan, PSI, gp, gp_len, wp = self.try_get_path(max_tries, los) # Get waypoint angle
        if tries == max_tries: # If path is not found
            return velocity, np.zeros(2), costmap_np, -1, 1 #-100

        angle = np.zeros([2])
        angle[0] = np.sin(angle_atan)
        angle[1] = np.cos(angle_atan)
        rotated_costmap, costmap_angle = self.add_angle_to_costmap(costmap_np, angle, PSI)
        if (np.max(rotated_costmap) - np.min(rotated_costmap)) != 0.0:
             rotated_costmap = (rotated_costmap - np.min(rotated_costmap)) / (np.max(rotated_costmap) - np.min(rotated_costmap))
        observation[1] = rotated_costmap

        # if self.check_is_stuck(action): # If stuck into wall
        #     print("STUCK!!!!!!!!!!!!!!!!!!!")

        #     return angle, obs, -50, 1 #-100

        # Episode finished
        if is_done == 1:
            reward = 10
        if self.status_move_base == 4: # 4 stands for goal aborted
            is_done = 1
            reward = -1 #-100
        if self.status_move_base == 3: # 3 stands for goal reached
            is_done = 1
            reward = 10

        if is_done:
            print(f"Distance to goal: {self.dist_to_goal()}")

        #return obs, reward, is_done
        #return wp, costmap_np, reward, is_done
        #normalized_wp = (wp + 1.75)/3.5
        #return wp, observation, reward, is_done
        #return orientation, wp, observation, reward, is_done
            
        return velocity, wp, rotated_costmap, reward, is_done
        #return velocity, wp, observation, reward, is_done

    def step_cdrl_short_wp(self, action):
        """ Update hidden given action, set action,  wait for rewards, return obs and is_done
         """ 
        ####################################
        ############ CLEARING  #############

        # Clear buffers
        self.clear_min_scan_buffer_size()
        self.clear_odom_buffer_size()

        ## Clear costmap every 0.5 seconds #
        self.clear_costmap_counter += 1
        if self.clear_costmap_counter == 10: # Before 10
            self.clear_costmap()
            self.clear_costmap_counter = 0
            
        ####################################
        ########### INITIALIZATION #########

        is_done = 0
        reward = 0
        velocity = np.zeros(2)
        observation = np.zeros([2,120,120])

        ####################################
        ########### GET STATE ##############
        
        # Costmap, waypoint, angle to waypoint
        obs_costmap = self.get_costmap()
        costmap_np = self.costmap_to_np(obs_costmap)
        max_tries = 5
        los = 1.75
        tries, angle_atan, PSI, gp, gp_len, wp = self.try_get_path(max_tries, 1.75) # Get waypoint angle
        if tries == max_tries: # If path is not found
            return np.zeros(2), np.zeros(2), costmap_np, -1, 1 #-100
        
        # Rotated costmap, costmap with waypoint
        angle = np.zeros([2])
        angle[0] = np.sin(angle_atan)
        angle[1] = np.cos(angle_atan)
        rotated_costmap, costmap_np_angle = self.add_angle_to_costmap(costmap_np, angle, PSI)
        if (np.max(rotated_costmap) - np.min(rotated_costmap)) != 0.0:
            rotated_costmap = (rotated_costmap - np.min(rotated_costmap)) / (np.max(rotated_costmap) - np.min(rotated_costmap))
        observation[0] = rotated_costmap

        # DWA action
        dwa_action = self.get_cmd_vel_dwa()

        ####################################
        ########### DO ACTION ##############

        # Take action
        activate_supervisor = False
        #supervisor_fuzzy, supervisor_general, positive = self.supervisor_fuzzy_v3([0.7757848524623762, 0.1472442306842174, 0.6331694604321398, 0.29670899571298637])

        if activate_supervisor:
            # Count how many times the supervisor is activated
            if self.supervisor_counter >= 1:
                # If no general add +1 to the counter
                self.supervisor_counter += 1
                self.supervisor_counter = (self.supervisor_counter % self.max_supervisor)
            if supervisor_general:
                # If general reset counter
                self.supervisor_counter = 1

            if (self.supervisor_counter != 0) or (supervisor_fuzzy == 1):
                if supervisor_general:
                    if positive:
                        #print("positive")
                        self.send_vel(np.asarray([-0.5, 0]))
                    else:
                        #print("negative")
                        self.send_vel(np.asarray([0.5, 0]))
                else:
                    #print("HERE!!!!!")
                    action_pursuit = self.pure_pursuit()
                    self.send_vel(action_pursuit)
            else:
                #print("supervised action")
                self.send_vel(action)
        else:
            #print("normal action")
            self.send_vel(action)
            #pass

        for i in range(1): #5
            self.rate_task.sleep()

        ####################################
        ########### GET REWARD #############

        # Get reward if reached waypoint
        self.wp_counter += 1
        X, Y = self.get_robot_position()
        wp_error = 1
        if (self.get_dist(np.array([X, Y]), self.wp_goal) < wp_error) and (self.get_dist(self.wp_goal, np.array([self.goal.position.x, self.goal.position.y])) > wp_error) or (self.wp_counter >= 50):
            #print(f"sub goal: {self.wp_goal}")
            reward = 10 # Before 10
            _, wp = self.get_dist_and_wp()
            if isinstance(wp, bool):
                return np.zeros(2), np.zeros(2), costmap_np, -1, 1  
            self.wp_goal = wp
            if self.wp_counter >= 50:
                #print("TIMEOUT")
                self.wp_counter = 0

        # Punish collision and time
        odom = self.get_odom()
        linear_x = odom.twist.twist.linear.x
        angular_z = odom.twist.twist.angular.z
        all_reward = self.reward_function_cdrl_short(0, linear_x)
        reward += all_reward
        is_done = self.goal_reached_dist()

        ####################################
        ########### GET NEXT STATE #########
   
        # Costmap, waypoint, angle to waypoint
        X, Y, Z, PSI, qt = self.get_robot_status()
        velocity[0] = linear_x
        velocity[1] = angular_z
        obs_costmap = self.get_costmap()
        costmap_np = self.costmap_to_np(obs_costmap)
        tries, angle_atan, PSI, gp, gp_len, wp = self.try_get_path(max_tries, los) # Get waypoint angle
        if tries == max_tries: # If path is not found
            return velocity, np.zeros(2), costmap_np, -1, 1 #-100

        # Rotated costmap, costmap with waypoint
        angle = np.zeros([2])
        angle[0] = np.sin(angle_atan)
        angle[1] = np.cos(angle_atan)
        rotated_costmap, costmap_angle = self.add_angle_to_costmap(costmap_np, angle, PSI)
        if (np.max(rotated_costmap) - np.min(rotated_costmap)) != 0.0:
             rotated_costmap = (rotated_costmap - np.min(rotated_costmap)) / (np.max(rotated_costmap) - np.min(rotated_costmap))
        observation[1] = rotated_costmap

        ####################################
        ########### EPISODE TERMINATION ####

        # Episode finished
        if is_done == 1:
            reward = 10
        if self.status_move_base == 4: # 4 stands for goal aborted
            is_done = 1
            reward = -1 #-100
        if self.status_move_base == 3: # 3 stands for goal reached
            is_done = 1
            reward = 10

        if is_done:
            print(f"Distance to goal: {self.dist_to_goal()}")
            
        return velocity, wp, rotated_costmap, reward, is_done

    def step_cdrl_short_test_wp(self, action):
        """ Update hidden given action, set action,  wait for rewards, return obs and is_done
         """ 
        ####################################
        ############ CLEARING  #############

        # Clear buffers
        self.clear_min_scan_buffer_size()
        self.clear_odom_buffer_size()

        ## Clear costmap every 0.5 seconds #
        self.clear_costmap_counter += 1
        if self.clear_costmap_counter == 10: # Before 10
            self.clear_costmap()
            self.clear_costmap_counter = 0
            
        ####################################
        ########### INITIALIZATION #########

        is_done = 0
        reward = 0
        velocity = np.zeros(2)
        observation = np.zeros([2,120,120])

        ####################################
        ########### GET STATE ##############
        
        # Costmap, waypoint, angle to waypoint
        obs_costmap = self.get_costmap()
        costmap_np = self.costmap_to_np(obs_costmap)
        max_tries = 5
        los = 1.75
        tries, angle_atan, PSI, gp, gp_len, wp = self.try_get_path(max_tries, 1.75) # Get waypoint angle
        if tries == max_tries: # If path is not found
            return 0, 0, np.zeros(2), np.zeros(2), costmap_np, -1, 1 #-100
        
        # Rotated costmap, costmap with waypoint
        angle = np.zeros([2])
        angle[0] = np.sin(angle_atan)
        angle[1] = np.cos(angle_atan)
        rotated_costmap, costmap_np_angle = self.add_angle_to_costmap(costmap_np, angle, PSI)
        if (np.max(rotated_costmap) - np.min(rotated_costmap)) != 0.0:
            rotated_costmap = (rotated_costmap - np.min(rotated_costmap)) / (np.max(rotated_costmap) - np.min(rotated_costmap))
        observation[0] = rotated_costmap

        # DWA action
        dwa_action = self.get_cmd_vel_dwa()

        ####################################
        ########### DO ACTION ##############

        # Take action
        activate_supervisor = True
        #supervisor_fuzzy, supervisor_general, positive = self.supervisor_fuzzy_v3([0.7757848524623762, 0.1472442306842174, 0.6331694604321398, 0.29670899571298637])
        supervisor_fuzzy, supervisor_general, positive = self.supervisor_fuzzy_v3([0.9184786224305393, 0.8189962593173226, 0.41726010041113815, 0.43826893116271404])

        if activate_supervisor:
            # Count how many times the supervisor is activated
            if self.supervisor_counter >= 1:
                # If no general add +1 to the counter
                self.supervisor_counter += 1
                self.supervisor_counter = (self.supervisor_counter % self.max_supervisor)
            if supervisor_general:
                # If general reset counter
                self.supervisor_counter = 1

            if (self.supervisor_counter != 0) or (supervisor_fuzzy == 1):
                if supervisor_general:
                    if positive:
                        #print("positive")
                        self.send_vel(np.asarray([-0.5, 0]))
                    else:
                        #print("negative")
                        self.send_vel(np.asarray([0.5, 0]))
                else:
                    #print("HERE!!!!!")
                    action_pursuit = self.pure_pursuit()
                    self.send_vel(action_pursuit)
            else:
                #print("supervised action")
                self.send_vel(action)
        else:
            #print("normal action")
            self.send_vel(action)

        #for i in range(5): #5
        self.rate_task.sleep()

        ####################################
        ########### GET REWARD #############

        # Get reward if reached waypoint
        self.wp_counter += 1
        X, Y = self.get_robot_position()
        wp_error = 1
        if (self.get_dist(np.array([X, Y]), self.wp_goal) < wp_error) and (self.get_dist(self.wp_goal, np.array([self.goal.position.x, self.goal.position.y])) > wp_error) or (self.wp_counter >= 50):
            print(f"sub goal: {self.wp_goal}")
            reward = 10 # Before 10
            _, wp = self.get_dist_and_wp()
            if isinstance(wp, bool):
                return 0, 0, np.zeros(2), np.zeros(2), costmap_np, -1, -1  
            self.wp_goal = wp
            if self.wp_counter >= 50:
                print("TIMEOUT")
                self.wp_counter = 0

        # Punish collision and time
        odom = self.get_odom()
        linear_x = odom.twist.twist.linear.x
        angular_z = odom.twist.twist.angular.z
        all_reward, collision_reward = self.reward_function_cdrl_short(0, linear_x)
        reward += all_reward
        is_done = self.goal_reached_dist()

        ####################################
        ########### GET NEXT STATE #########
   
        # Costmap, waypoint, angle to waypoint
        X, Y, Z, PSI, qt = self.get_robot_status()
        velocity[0] = linear_x
        velocity[1] = angular_z
        obs_costmap = self.get_costmap()
        costmap_np = self.costmap_to_np(obs_costmap)
        tries, angle_atan, PSI, gp, gp_len, wp = self.try_get_path(max_tries, los) # Get waypoint angle
        if tries == max_tries: # If path is not found
            return 0, supervisor_general, velocity, np.zeros(2), costmap_np, -1, -1 #-100

        # Rotated costmap, costmap with waypoint
        angle = np.zeros([2])
        angle[0] = np.sin(angle_atan)
        angle[1] = np.cos(angle_atan)
        rotated_costmap, costmap_angle = self.add_angle_to_costmap(costmap_np, angle, PSI)
        if (np.max(rotated_costmap) - np.min(rotated_costmap)) != 0.0:
             rotated_costmap = (rotated_costmap - np.min(rotated_costmap)) / (np.max(rotated_costmap) - np.min(rotated_costmap))
        observation[1] = rotated_costmap

        ####################################
        ########### EPISODE TERMINATION ####

        # Episode finished
        if is_done == 1:
            reward = 10
        if self.status_move_base == 4: # 4 stands for goal aborted
            is_done = 1
            reward = -1 #-100
        if self.status_move_base == 3: # 3 stands for goal reached
            is_done = 1
            reward = 10

        if is_done:
            print(f"Distance to goal: {self.dist_to_goal()}")
            
        return collision_reward, supervisor_general, velocity, wp, rotated_costmap, reward, is_done

    def set_wp_goal(self):
        X, Y = self.get_robot_position()
        self.wp_goal = np.array([X, Y])

    
    def step_cdrl_train_supervisor(self, action, x):
        # Update hidden given action, set action,  wait for rewards, return obs and is_done
        # clear buffers
        self.clear_min_scan_buffer_size()
        self.clear_odom_buffer_size()
        # Clear costmap every 0.5 seconds
        self.clear_costmap_counter += 1
        if self.clear_costmap_counter == 10:
            self.clear_costmap()
            self.clear_costmap_counter = 0

        # Initialization
        is_done = 0
        supervisor_fuzzy_counter = 0
        supervisor_general_fuzzy_counter = 0

        orientation = np.zeros(2)
        observation = np.zeros([2,60,60])
        
        # Get state
        obs_costmap = self.get_costmap()
        costmap_np = self.costmap_to_np(obs_costmap)
        max_tries = 5
        los = 1.75
        tries, angle_atan, PSI, gp, gp_len, wp = self.try_get_path(max_tries, los) # Get waypoint angle
        if tries == max_tries: # If path is not found
            return 0, 0, np.zeros(2), np.zeros(2), costmap_np, -1, 1 #-100
        
        angle = np.zeros([2])
        angle[0] = np.sin(angle_atan)
        angle[1] = np.cos(angle_atan)
        rotated_costmap, costmap_np_angle = self.add_angle_to_costmap(costmap_np, angle, PSI)
        if (np.max(rotated_costmap) - np.min(rotated_costmap)) != 0.0:
            rotated_costmap = (rotated_costmap - np.min(rotated_costmap)) / (np.max(rotated_costmap) - np.min(rotated_costmap))
        observation[0] = costmap_np

        dwa_action = self.get_cmd_vel_dwa()

        init_point, wp = self.get_dist_and_wp()
        if isinstance(wp, bool):
            return 0, 0, np.zeros(2), np.zeros(2), costmap_np, -1, 1  

        # Take action
        activate_supervisor = True
        if activate_supervisor:
            self.supervisor_counter, supervisor_general, positive = self.supervisor_fuzzy_v3(x)
            # Count how many times the supervisor is activated
            if self.supervisor_counter == 1 and not supervisor_general:
                supervisor_fuzzy_counter = 1
            if supervisor_general:
                supervisor_general_fuzzy_counter = 1 

            if self.supervisor_counter != 0:
                if supervisor_general:
                    if positive:
                        self.send_vel(np.asarray([-0.2, 0]))
                    else:
                        self.send_vel(np.asarray([0.2, 0]))
                else:
                    self.send_vel(dwa_action)
                if self.supervisor_counter == 1:
                    #self.send_vel(np.asarray([0,0]))
                    pass
                    #self.send_vel(dwa_action)
            else:
                #print("there")
                self.send_vel(action)
        else:
            #print("there")
            self.send_vel(action)
        for i in range(5): #5
            self.rate_task.sleep()

        # Get reward
        #reward  = self.reward_function_cdrl(costmap_np_angle, action, dwa_action)
        traveled_dist = self.get_dist_traveled(init_point, wp)
        odom = self.get_odom()
        linear_x = odom.twist.twist.linear.x
        angular_z = odom.twist.twist.angular.z
        reward = self.reward_function_cdrl_short(traveled_dist)
        is_done = self.goal_reached_dist()
   

        # Get next state
        X, Y, Z, PSI, qt = self.get_robot_status()
        orientation[0] = np.sin(PSI)
        orientation[1] = np.cos(PSI)
        obs_costmap = self.get_costmap()
        costmap_np = self.costmap_to_np(obs_costmap)
        tries, angle_atan, PSI, gp, gp_len, wp = self.try_get_path(max_tries, los) # Get waypoint angle
        if tries == max_tries: # If path is not found
            return supervisor_fuzzy_counter, supervisor_general_fuzzy_counter, orientation, np.zeros(2), costmap_np, -1, 1 #-100

        angle = np.zeros([2])
        angle[0] = np.sin(angle_atan)
        angle[1] = np.cos(angle_atan)
        rotated_costmap, costmap_angle = self.add_angle_to_costmap(costmap_np, angle, PSI)
        if (np.max(rotated_costmap) - np.min(rotated_costmap)) != 0.0:
             rotated_costmap = (rotated_costmap - np.min(rotated_costmap)) / (np.max(rotated_costmap) - np.min(rotated_costmap))
        observation[1] = costmap_np

        # if self.check_is_stuck(action): # If stuck into wall
        #     print("STUCK!!!!!!!!!!!!!!!!!!!")

        #     return angle, obs, -50, 1 #-100

        # Episode finished
        if is_done == 1:
            reward = 10
        if self.status_move_base == 4: # 4 stands for goal aborted
            is_done = 1
            reward = -1 #-100
        if self.status_move_base == 3: # 3 stands for goal reached
            is_done = 1
            reward = 10

        #if is_done:
            #print(f"Distance to goal: {self.dist_to_goal()}")

        #return obs, reward, is_done
        #return wp, costmap_np, reward, is_done
        #normalized_wp = (wp + 1.75)/3.5
        #return wp, observation, reward, is_done
        #return orientation, wp, observation, reward, is_done
        return supervisor_fuzzy_counter, supervisor_general_fuzzy_counter, orientation, wp, rotated_costmap, reward, is_done


    def step_cdrl_train_supervisor_wp(self, action, x):
        """ Update hidden given action, set action,  wait for rewards, return obs and is_done
         """ 
        ####################################
        ############ CLEARING  #############

        # Clear buffers
        self.clear_min_scan_buffer_size()
        self.clear_odom_buffer_size()

        ## Clear costmap every 0.5 seconds #
        self.clear_costmap_counter += 1
        if self.clear_costmap_counter == 10: # Before 10
            self.clear_costmap()
            self.clear_costmap_counter = 0

        ####################################
        ########### INITIALIZATION #########

        is_done = 0
        reward = 0
        velocity = np.zeros(2)
        observation = np.zeros([2,120,120])

        # Fuzzy
        supervisor_fuzzy_counter = 0
        supervisor_general_fuzzy_counter = 0

        ####################################
        ########### GET STATE ##############
        
        # Costmap, waypoint, angle to waypoint
        obs_costmap = self.get_costmap()
        costmap_np = self.costmap_to_np(obs_costmap)
        max_tries = 5
        los = 1.75
        tries, angle_atan, PSI, gp, gp_len, wp = self.try_get_path(max_tries, 1.75) # Get waypoint angle
        if tries == max_tries: # If path is not found
            return 0, 0, np.zeros(2), np.zeros(2), costmap_np, -1, 1 #-100
        
        # Rotated costmap, costmap with waypoint
        angle = np.zeros([2])
        angle[0] = np.sin(angle_atan)
        angle[1] = np.cos(angle_atan)
        rotated_costmap, costmap_np_angle = self.add_angle_to_costmap(costmap_np, angle, PSI)
        if (np.max(rotated_costmap) - np.min(rotated_costmap)) != 0.0:
            rotated_costmap = (rotated_costmap - np.min(rotated_costmap)) / (np.max(rotated_costmap) - np.min(rotated_costmap))
        observation[0] = rotated_costmap

        # DWA action
        dwa_action = self.get_cmd_vel_dwa()

        ####################################
        ########### DO ACTION ##############

        # Take action
        activate_supervisor = True
        if activate_supervisor:
            supervisor_fuzzy, supervisor_general, positive = self.supervisor_fuzzy_v3(x)
            # Count how many times the supervisor is activated
            if self.supervisor_counter >= 1:
                # If no general add +1 to the counter
                self.supervisor_counter += 1
                self.supervisor_counter = (self.supervisor_counter % self.max_supervisor)
            if supervisor_general:
                # If general reset counter
                self.supervisor_counter = 1

            if (self.supervisor_counter != 0) or (supervisor_fuzzy == 1):
                if supervisor_general:
                    if positive:
                        self.send_vel(np.asarray([-0.9, 0]))
                    else:
                        self.send_vel(np.asarray([0.9, 0]))
                else:
                    #print("HERE!!!!!")
                    action_pursuit = self.pure_pursuit()
                    self.send_vel(action_pursuit)
            else:
                #print("there")
                self.send_vel(action)
        else:
            #print("there")
            self.send_vel(action)
        for i in range(5): #5
            self.rate_task.sleep()

        ####################################
        ########### GET REWARD #############

        # Get reward if reached waypoint
        self.wp_counter += 1
        X, Y = self.get_robot_position()
        wp_error = 1
        if (self.get_dist(np.array([X, Y]), self.wp_goal) < wp_error) and (self.get_dist(self.wp_goal, np.array([self.goal.position.x, self.goal.position.y])) > wp_error) or (self.wp_counter >= 50):
            print(f"sub goal: {self.wp_goal}")
            reward = 10
            _, wp = self.get_dist_and_wp()
            if isinstance(wp, bool):
                return supervisor_fuzzy, (1 if self.supervisor_counter >= 1 else 0), np.zeros(2), np.zeros(2), costmap_np, -1, 1  
            self.wp_goal = wp
            if self.wp_counter >= 50:
                print("TIMEOUT")
                self.wp_counter = 0

        # Punish collision and time
        odom = self.get_odom()
        linear_x = odom.twist.twist.linear.x
        angular_z = odom.twist.twist.angular.z
        reward += self.reward_function_cdrl_short(0, linear_x)
        is_done = self.goal_reached_dist()

        ####################################
        ########### GET NEXT STATE #########
   
        # Costmap, waypoint, angle to waypoint
        X, Y, Z, PSI, qt = self.get_robot_status()
        velocity[0] = linear_x
        velocity[1] = angular_z
        obs_costmap = self.get_costmap()
        costmap_np = self.costmap_to_np(obs_costmap)
        tries, angle_atan, PSI, gp, gp_len, wp = self.try_get_path(max_tries, los) # Get waypoint angle
        if tries == max_tries: # If path is not found
            return supervisor_fuzzy, (1 if self.supervisor_counter >= 1 else 0), velocity, np.zeros(2), costmap_np, -1, 1 #-100

        # Rotated costmap, costmap with waypoint
        angle = np.zeros([2])
        angle[0] = np.sin(angle_atan)
        angle[1] = np.cos(angle_atan)
        rotated_costmap, costmap_angle = self.add_angle_to_costmap(costmap_np, angle, PSI)
        if (np.max(rotated_costmap) - np.min(rotated_costmap)) != 0.0:
             rotated_costmap = (rotated_costmap - np.min(rotated_costmap)) / (np.max(rotated_costmap) - np.min(rotated_costmap))
        observation[1] = rotated_costmap
        
        ####################################
        ########### EPISODE TERMINATION ####

        # Episode finished
        if is_done == 1:
            reward = 10
        if self.status_move_base == 4: # 4 stands for goal aborted
            is_done = 1
            reward = -1 #-100
        if self.status_move_base == 3: # 3 stands for goal reached
            is_done = 1
            reward = 10

        print(f"self.supervisor_counter: {self.supervisor_counter}")
        print(f"supervisor_fuzzy: {supervisor_fuzzy}")
        print(f"supervisor_general: {supervisor_general}")
        return supervisor_fuzzy, (1 if self.supervisor_counter >= 1 else 0), velocity, wp, rotated_costmap, reward, is_done


    def step_vae_mdrnn_feedback(self, action, action_num, rl_distribution):
        # Step for vae and mdrnn
        # Update hidden given action, set action,  wait for rewards, return obs and is_done

        # Ablation
        ablation = 2 # 0 Just lidar, 1 just VAE

        # clear buffers
        self.clear_min_scan_buffer_size()
        self.clear_odom_buffer_size()
        # Clear costmap every 0.5 seconds
        self.clear_costmap_counter += 1
        if self.clear_costmap_counter == 10:
            self.clear_costmap()
            self.clear_costmap_counter = 0

        # Initialization
        is_done = 0
        action = action
        if self.discrete_actions == False:
            action = self.translate_continous_action(action)
        
        # Update hidden
        obs_costmap = self.get_costmap()
        latent_mu, logvar_i_o = self.process_obs_3(obs_costmap)
        mdrnn_action = self.get_action_for_mdrnn(action)
        with torch.no_grad():
            _, _, _, _, _, self.hidden = self.mdrnn(mdrnn_action, latent_mu, self.hidden)


        # Take action
        self.tune_parameters(action)
        self.rate_task.sleep()

        # Get observations
        lidar = self.get_filtered_scan()
        obs_costmap = self.get_costmap()
        latent_mu, _ = self.process_obs_3(obs_costmap)
        vel = self.get_vel()
        obs = self.extract_3(latent_mu, self.hidden[0], vel)
        obs_0 = self.extract_lidar(lidar, vel)
        obs_1 = self.extract_vae(latent_mu, vel)

        # Set reward
        reward = self.reward_feedback(obs, action_num, rl_distribution)

        # Episode finished
        if self.status_move_base == 4: # 4 stands for goal aborted
            is_done = 1
            reward = -10
        if self.status_move_base == 3: # 3 stands for goal reached
            is_done = 1
            reward = 0

        if ablation == 0:
            return obs_0, reward, is_done
        elif ablation == 1:
            return obs_1, reward, is_done
        else:
            return obs, reward, is_done
    
    def step_applr(self, action):
        # Step for applr
        # Update hidden given action, set action,  wait for rewards, return obs and is_done

        # clear buffers
        self.clear_min_scan_buffer_size()
        self.clear_odom_buffer_size()
        # Clear costmap every 0.5 seconds
        self.clear_costmap_counter += 1
        if self.clear_costmap_counter == 10:
            self.clear_costmap()
            self.clear_costmap_counter = 0

        # Initialization
        is_done = 0
        if self.discrete_actions == False:
            action_param = self.translate_continous_action(action, True)
        
        # Update hidden
        obs_costmap = self.get_costmap()
        latent_mu, logvar_i_o = self.process_obs_3(obs_costmap)
        mdrnn_action = self.get_action_for_mdrnn(action_param)
        with torch.no_grad():
            _, _, _, _, _, self.hidden = self.mdrnn(mdrnn_action, latent_mu, self.hidden)


        # Take action
        self.tune_parameters(action_param)
        self.rate_task.sleep()

        # Get observations
        lidar = self.get_filtered_scan()
        obs_costmap = self.get_costmap()
        latent_mu, _ = self.process_obs_3(obs_costmap)
        vel = self.get_vel()
        normalized_vel = self.normalize_vel(vel)
        obs_0 = self.extract_applr(lidar, action, vel)
        obs_1 = self.extract_3(latent_mu, self.hidden[0], vel)
        obs_2 = self.extract_lidar(lidar, vel)
        obs_3 = self.extract_vae(latent_mu, vel)

        # Set reward
        reward = self.reward_func(lidar)

        # Episode finished
        if self.status_move_base == 4: # 4 stands for goal aborted
            is_done = 1
            reward = -10
        if self.status_move_base == 3: # 3 stands for goal reached
            is_done = 1
            reward = 0

        return obs_0, reward, is_done

    def step_vae_phydnet(self, action):
        # Step for vae and phydnet
        # Set action,  wait for rewards, obs and is_done
        self.clear_min_scan_buffer_size()
        self.clear_odom_buffer_size()
        # Clear costmap every 0.5 seconds
        self.clear_costmap_counter += 1
        if self.clear_costmap_counter == 10:
            self.clear_costmap()
            self.clear_costmap_counter = 0

        # Initialization
        is_done = 0
        action = action
        if self.discrete_actions == False:
            action = self.translate_continous_action(action, False)
        
        # Take action
        self.tune_parameters(action)
        self.rate_task.sleep()

        # Get observations
        lidar = self.get_filtered_scan()
        costmap_buffer = self.get_costmap_buffer()
        latent_sequence = self.get_latent_phydnet(costmap_buffer)
        vel = self.get_vel()
        obs = self.extract_4(latent_sequence, vel)

        # Set reward
        reward = self.reward_func(lidar)

        # Episode finished
        if self.status_move_base == 4: # 4 stands for goal aborted
            is_done = 1
            reward = -10
        if self.status_move_base == 3: # 3 stands for goal reached
            is_done = 1
            reward = 0

        return obs, reward, is_done


    def downsample(scan):
        len_scan = len(scan)
        desired_len = 720
        rate = len_scan/desired_len
        new_scan = []
        for i in range(720):
            new_scan[i] = scan[int(math.floor(rate*i))]
        return np.array(new_scan)
    
    def extract(self, obs, prediction):

        obs = obs[:,-1,:]
        prediction_mu = prediction[0]
        prediction_sigma = prediction[1]

        prediction_mu = prediction_mu.detach()
        prediction_mu = prediction_mu.numpy()
        prediction_mu = prediction_mu[:,-1,:]

        prediction_sigma = prediction_sigma.detach()
        prediction_sigma = prediction_sigma.numpy()
        prediction_sigma = prediction_sigma[:,-1,:]

        obs_cat = np.concatenate((obs[0], prediction_mu[0], prediction_sigma[0]), axis = 0)

        return (obs_cat)
    
    def extract_2(self, mu_i_o, logvar_i_o, mu_p_o, logvar_p_o):
        mu_i_o = mu_i_o.cpu().data.numpy()[0]
        logvar_i_o = logvar_i_o.cpu().data.numpy()[0]
        mu_p_o = mu_p_o.cpu().data.numpy()[0]
        logvar_p_o = logvar_p_o.cpu().data.numpy()[0]

        obs_cat = np.concatenate((mu_i_o, logvar_i_o, mu_p_o, logvar_p_o), axis = 0)

        return (obs_cat)
    
    def extract_3(self, mu_i_o, logvar_i_o, vel):
        #mu_i_o = mu_i_o.cpu().data.numpy()[0]
        #logvar_i_o = logvar_i_o.cpu().data.numpy()[0]
        # with torch.no_grad():
        #     vel = torch.cuda.FloatTensor(vel)
        #     mu_i = torch.squeeze(mu_i_o)
        #     logvar_i = torch.squeeze(logvar_i_o)
        #     obs_cat = torch.cat((mu_i, logvar_i, vel), 0)
        #     #obs_cat = obs_cat.cpu().data.numpy()
        #     print(type(obs_cat))

        #     return (obs_cat)
        mu_i_o = mu_i_o.cpu().data.numpy()[0]
        logvar_i_o = logvar_i_o.cpu().data.numpy()[0]

        obs_cat = np.concatenate((mu_i_o, logvar_i_o, vel), axis = 0)

        return (obs_cat)
    
    def extract_lidar(self, lidar, vel):
        obs_cat = np.concatenate((lidar, vel), axis = 0)

        return (obs_cat)

    def extract_vae(self, mu, vel):
        mu_np = mu.cpu().data.numpy()[0]

        obs_cat = np.concatenate((mu_np, vel), axis = 0)

        return (obs_cat)

    def extract_applr(self, lidar, action, vel):
        obs_cat = np.concatenate((lidar, action, vel), axis = 0)


        return (obs_cat)
    
    def extract_4(self, list_tensors, vel):
        obs_cat = np.zeros([0])
        for i in range(len(list_tensors)):
            np_ten = list_tensors[i].cpu().data.numpy()
            obs_cat = np.concatenate((obs_cat, np_ten), axis = 0)

        obs_cat = np.concatenate((obs_cat, vel), axis = 0)

        return (obs_cat)
    
    def normalize_vel(self, vel):
        normalized_vel = vel/np.sqrt(vel[0]**2 + vel[1]**2)
        return normalized_vel


    def proccess_rewards(self, rewards):
        # If hitted return hitted, else return reached, else retun time penalty.

        rewards.sort()
        if rewards[0] != -0.005:
            return rewards[0]
        return rewards[-1]

    def proccess_rewards_2(self, rewards):
        # If hitted return hitted, else return distance covered.
        
        rewards.sort()
        if rewards[0] <= -50: # == -40
            #print("Hitted obstacle!!!")
            print("here")
            return rewards[0]
        return sum(rewards)

    def proccess_rewards_3(self, rewards):
        # Sum all rewards
        return sum(rewards)

    def proccess_is_dones(self, is_dones):
        is_dones.sort()
        if is_dones[-1] == 1:
            return True
        return False
    
    def process_obs(self, obs):
        # Get x and y pos of each marker, append the last n frames. Predcit and return last
        # input frame, and the last predicted frame.
        np_obs = np.zeros([1,self.num_obs_for_prediction,(self.num_tracks)*2])

        if len(obs) >= self.num_obs_for_prediction:
            num_obs = self.num_obs_for_prediction
        else:
            num_obs = len(obs)

        for iter in range(len(obs) - num_obs, len(obs),1):
            i = iter - len(obs) + num_obs
            for track in range(self.num_tracks):
                np_obs[:, i, track*2] = obs[i].markers[track].pose.position.x
                np_obs[:, i, track*2 + 1] = obs[i].markers[track].pose.position.y
        
        for iter in range(self.num_obs_for_prediction - num_obs):
            i = iter + num_obs
            for track in range(self.num_tracks):
                np_obs[:, i, track*2] = obs[-1].markers[track].pose.position.x
                np_obs[:, i, track*2 + 1] = obs[-1].markers[track].pose.position.y
        
        prediction = self.prediction_module.predict(np_obs)

        return np_obs, prediction

    def costmap_to_np(self, costmap):
        # Occupancy grid to numpy
        #costmap_np = np.zeros([60,60])
        costmap_np = np.zeros([120,120])
        if costmap is not None:
            #costmap_np = np.divide(np.array_split(costmap.data, 60),100) # costmap size is 60
            costmap_np = np.divide(np.array_split(costmap.data, 120),100) # costmap size is 60
        return costmap_np
    
    def phydnet_get_encoder_hidden(self, costmap_sequence):
        # Get data of costmap buffer. Return the encoder_hidden of phydnet.
        video = np.zeros([1,10,1,60,60])
        for itr in range(self.n_frames):
            video[0,itr,0,:,:] = self.costmap_to_np(costmap_sequence[itr])
        encoder_hidden = self.phydnet.get_encoding(video)

        return encoder_hidden
    
    def get_latent_phydnet(self, costmap_buffer):
        # Get data of costmap buffer. Return the VAE encoding of the last 5 frames, and the VAE encoding
        # of the phydnet prediction of the next 5 frames. Latent of phydnet is too big.
        images = np.zeros([10,3,64,64])
        video = np.zeros([1,10,1,60,60])
        for itr in range(self.n_frames):
            video[0,itr,0,:,:] = self.costmap_to_np(costmap_buffer[itr])

        predictions = self.phydnet.predict(video)

        # Concatenate the last 5 frames of costmap_buffer
        for i_frame in range(0,5):
            for i_channel in range(self.img_channels):
                images[i_frame,i_channel,0:60,0:60] = self.costmap_to_np(costmap_buffer[5 + i_frame])
        # Concatenate the predicted next 5 frames
        for i_frame in range(5,10):
            for i_channel in range(self.img_channels):
                images[i_frame,i_channel,0:60,0:60] = np.squeeze(predictions[:,i_frame - 5,:,:])
        
        # Encode everything using VAE
        _, latents, _ = self.vae_input_inflation(torch.cuda.FloatTensor(images))
        latents = latents.reshape(-1)

        #####################################################
        # Show videos #######################################
        # fig, (ax1, ax2) = plt.subplots(1, 2)
        # fig.suptitle('Videos')

        # #print("Input video")
        # frames1 = [] # for storing the generated images
        # for i in range(5):
        #     frames1.append([ax1.imshow(np.fliplr(np.squeeze(video[0,i+5,0,:,:])), cmap=cm.Greys_r,animated=True)])
        # ax1.set_title("Input")

        # #print("Prediction video")
        # frames2 = [] # for storing the generated images
        # for i in range(5):
        #     frames2.append([ax2.imshow(np.fliplr(np.squeeze(predictions[0,i,:,:])), cmap=cm.Greys_r,animated=True)])
        # ax2.set_title("Predicted")

        # ani = animation.ArtistAnimation(fig, frames1, interval=500, blit=True)
        # ani2 = animation.ArtistAnimation(fig, frames2, interval=500, blit=True)
        # plt.show(block=False)
        # plt.pause(5)
        # plt.close()
        # plt.show()
        
        return [latents]


    
    def process_obs_2(self, obs):
        # Get data of costmap buffer. Return the encoding of the last frame (the current costmap).
        # predict the next n frames. Return the encoding of the last frame.
        image_i = np.zeros([1,3,64,64]) # 3 channels. 64x64 costmap (instead of 60x60)
        video = np.zeros([1,10,1,60,60])
        image_p = np.zeros([1,3,64,64])

        for itr in range(self.img_channels):
            image_i[0,itr,0:60,0:60] = self.costmap_to_np(obs[-1])
          
        for itr in range(self.n_frames):
            video[0,itr,0,:,:] = self.costmap_to_np(obs[itr])
        
        predictions = self.phydnet.predict(video)
        for itr in range(self.img_channels):
            image_p[0,itr,0:60,0:60] = np.squeeze(predictions[:,-1,:,:])

        # Encode input image
        image_i = torch.tensor(image_i)
        image_i = image_i.to(self.device, dtype=torch.float)
        image_i_o, mu_i_o, logvar_i_o = self.vae_input(image_i)

        # Encode predicted image
        image_p = torch.tensor(image_p)
        image_p = image_p.to(self.device, dtype=torch.float)
        image_p_o, mu_p_o, logvar_p_o  = self.vae_prediction(image_p)

        #####################################################
        # Show videos #######################################
        # fig, (ax1, ax2) = plt.subplots(1, 2)
        # fig.suptitle('Videos')

        # #print("Input video")
        # frames1 = [] # for storing the generated images
        # for i in range(10):
        #     frames1.append([ax1.imshow(np.fliplr(np.squeeze(video[0,i,0,:,:])), cmap=cm.Greys_r,animated=True)])
        # ax1.set_title("Input")

        # #print("Prediction video")
        # frames2 = [] # for storing the generated images
        # for i in range(10):
        #     frames2.append([ax2.imshow(np.fliplr(np.squeeze(predictions[0,i,:,:])), cmap=cm.Greys_r,animated=True)])
        # ax2.set_title("Predicted")

        # ani = animation.ArtistAnimation(fig, frames1, interval=100, blit=True)
        # ani2 = animation.ArtistAnimation(fig, frames2, interval=100, blit=True)

        ####################################################
        ####################################################

        ####################################################
        #Show images #######################################

        # fig2, (ax11, ax22) = plt.subplots(1, 2)
        # fig2.suptitle('Images')

        # im1 = Image.fromarray(np.fliplr(np.squeeze(image_i_o[0,0].cpu().data.numpy()))*256)
        # imgplot = ax11.imshow(im1)
        # ax11.set_title("Decoded input")


        # im3 = Image.fromarray(np.fliplr(np.squeeze(image_p_o[0,0].cpu().data.numpy()))*256)
        # imgplot = ax22.imshow(im3)
        # ax22.set_title("Decoded prediction")

        #####################################################
        #####################################################

        #plt.show(block=False)
        #plt.pause(1)
        #plt.close()
        #plt.show()
        
        return mu_i_o, logvar_i_o, mu_p_o, logvar_p_o
        
    
    def process_obs_3(self, obs):
        # Encode costmap/s.
        image_i = np.zeros([1,3,64,64]) # 3 channels. 64x64 costmap (instead of 60x60)

        for itr in range(self.img_channels):
            image_i[0,itr,0:60,0:60] = self.costmap_to_np(obs)
          
        # Encode input image
        image_i = torch.tensor(image_i)
        image_i = image_i.to(self.device, dtype=torch.float)
        image_i_o, mu_i_o, logvar_i_o = self.vae_input_inflation(image_i)


        ####################################################
        #Show images #######################################

        # fig2, (ax11, ax22) = plt.subplots(1, 2)
        # fig2.suptitle('Images')

        # im1 = Image.fromarray(np.fliplr(np.squeeze(image_i_o[0,0].cpu().data.numpy()))*256)
        # imgplot = ax11.imshow(im1)
        # ax11.set_title("Input")


        # im3 = Image.fromarray(np.fliplr(np.squeeze(obs.cpu().data.numpy()))*256)
        # imgplot = ax22.imshow(im3)
        # ax22.set_title("Decoded input")

        #####################################################
        #####################################################

        # plt.show(block=False)
        # plt.pause(1)
        # plt.close()
        # plt.show()
        
        return mu_i_o, logvar_i_o

    def process_obs_4(self, obs):

        num_costmaps = 10
        vae_cat = np.zeros([0])
        for itr in range(num_costmaps):
            # Encode costmap/s.
            image_i = np.zeros([1,3,64,64]) # 3 channels. 64x64 costmap (instead of 60x60)

            for itr in range(self.img_channels):
                image_i[0,itr,0:60,0:60] = self.costmap_to_np(obs[itr])
            
            # Encode input image
            image_i = torch.tensor(image_i)
            image_i = image_i.to(self.device, dtype=torch.float)
            image_i_o, mu_i_o, logvar_i_o = self.vae_input_inflation(image_i)

            vae_cat = np.concatenate(vae_cat, mu_i_o, logvar_i_o)


        ####################################################
        #Show images #######################################

        # fig2, (ax11, ax22) = plt.subplots(1, 2)
        # fig2.suptitle('Images')

        # im1 = Image.fromarray(np.fliplr(np.squeeze(image_i_o[0,0].cpu().data.numpy()))*256)
        # imgplot = ax11.imshow(im1)
        # ax11.set_title("Input")


        # im3 = Image.fromarray(np.fliplr(np.squeeze(obs.cpu().data.numpy()))*256)
        # imgplot = ax22.imshow(im3)
        # ax22.set_title("Decoded input")

        #####################################################
        #####################################################

        # plt.show(block=False)
        # plt.pause(1)
        # plt.close()
        # plt.show()
        
        return vae_cat

    def is_stuck(self, action):
        # Check if robot is not moving

        odom = self.get_odom()
        if np.abs(odom.twist.twist.linear.x) < 0.005 and (np.abs(action[0]) > 0.05):
            #print(odom)
            #print(f"odom: {odom.twist.twist.linear}")
            print(action[0])
            print("stuck")
            return True
        return False

    def check_is_stuck(self, action):
        # Check if robot does not move for more than 3 seconds
        max_time = 5
        for i in range(max_time):
            if not self.is_stuck(action):
                return False
            self.rate_task.sleep()
        print(f"i: {i}")
        return True

    
    def try_get_path(self, max_tries, los):
        # Try to get path variables

        angle_atan, PSI, gp, gp_len, wp = self.get_local_goal_angle(los) 

        tries = 0
        for i in range(max_tries):
            tries += 1
            if isinstance(gp, bool) and gp is False:
                print("here 1")
                self.rate_task.sleep()
                angle_atan, PSI, gp, gp_len, wp = self.get_local_goal_angle(los) 
            else:
                break
            self.rate_task.sleep()

        return tries, angle_atan, PSI, gp, gp_len, wp
    
    def hit_obstacle(self):
        # Get minimum reading of scanner

        scann = self.get_scan()

        if min(scann.ranges) < self.min_dist_to_obstacle:
            #print("Min scan!!!")
            #print(min(scann.ranges))
            return True
        return False

    def hit_obstacle_2(self):
        # Get k minimum readings of scanner
        # k = 5 reafactor 10
        k = 5 # 5 real factor 40

        scann = self.get_scan()
        ranges = np.array(scann.ranges)
        idx = np.argpartition(ranges, k)
        ranges = ranges[idx[:k]]

        if np.max(ranges) < self.min_dist_to_obstacle:
            #print("Min scan!!!")
            #print(ranges)
            return True, np.max(ranges)
        return False, None

    def goal_reached(self):
        # Check if robot reached destination

        status_move_base = self.get_status_move_base()

        return status_move_base == 3
    
    def goal_reached_dist(self):
        """ Check if goal reached destination according to how close it is from the goal
        """
        min_dist = 0.5

        X, Y = self.get_robot_position()
        current_pos = np.array([X, Y])
        #print(f"goal: {self.goal.position.x, self.goal.position.y}")

        dist_to_goal = self.get_dist(current_pos, np.array([self.goal.position.x, self.goal.position.y]))
        
        if dist_to_goal < min_dist:
            return 1
        
        return 0
    
    def dist_to_goal(self):
        odom = self.get_odom()
        
        return np.sqrt((odom.pose.pose.position.x - self.goal.position.x + self.offset[0])**2 + (odom.pose.pose.position.y - self.goal.position.y + self.offset[1])**2)
    
    def reward_dist_to_goal(self):

        dist = self.dist_to_goal()
        return 2*(np.tanh(2 - dist/5))
    
    def get_covered_distance(self, pos1, pos2):
        # Get euclidean distance between two positions

        return np.sqrt((pos1.pose.pose.position.x - pos2.pose.pose.position.x)**2 + (pos1.pose.pose.position.y - pos2.pose.pose.position.y)**2)
    
    def adjust_covered_distance(self, dist):
        # Negative reward for distance covered. The more distance covered the smallest the reward. Between 0 and -1
        # Distance covered is usually between 0.2 and 0.02. Typically 0.09

        return (np.tanh(dist/0.31) - 1) #0.08 without multiplication
    
    def translate_continous_action(self, values, exact):
        # Translate from numbers to values of parameters for parameter tuning

        action = {}

        for itr in range(len(self.list_tune_params)):
            if exact:
                val = values[itr]
            else:
                val = self.get_action(self.list_tune_params[itr][1][0], self.list_tune_params[itr][1][1], values[itr])
            if self.list_tune_params[itr][0] != 'vel':
                action[self.list_tune_params[itr][0]] = val
            else:
                if self.local_planner == "dwa":
                    action['max_vel_x'] = val
                    action['min_vel_x'] = -val
                    action['max_vel_trans'] = val
                else:
                    action['max_vel_x'] = val
        return action

    
    def get_action(self, min, max, value):

        return min + (max - min)*((value + 1)/2)
    
    def hit_obstacle_reward(self, dist_to_obstacle):

        return -60*(0.4 - dist_to_obstacle)
        
    def adjust_hit_rewards(self, dist_to_obstacle):
        # Reward for proximity to obstacles.
        # Negative reward for distance covered. The more distance covered the smallest the reward. Between 0 and -1
        # Distance covered is usually between 0.2 and 0.02. Typically 0.09

        return 3*(np.tanh(dist_to_obstacle/0.25) - 1) # - Before 8
    
    def get_action_for_mdrnn(self, action):
        inflation = action['inflation_radius']
        cmd_vel = self.get_cmd_vel()
        mdrnn_action = np.zeros([1,3])
        mdrnn_action[0,0] = cmd_vel.linear.x
        mdrnn_action[0,1] = cmd_vel.angular.z
        mdrnn_action[0,2] = inflation

        return torch.cuda.FloatTensor(mdrnn_action)

    
    def get_filtered_scan(self):

        scan = self.get_scan()
        ranges = np.asarray(scan.ranges)
        for i in range(0, len(ranges)):
            if ranges[i] == inf:
                ranges[i] = 3.5

        return ranges
    
    def get_vel(self):
        odom = self.get_odom()
        vel = np.zeros([2])
        vel[0] = odom.twist.twist.linear.x
        vel[1] = odom.twist.twist.angular.z
        
        return vel
    
    def get_comand_vel(self):
        cmd_vel = self.get_cmd_vel()
        vel = np.zeros([2])
        vel[0] = cmd_vel.linear.x
        vel[1] = cmd_vel.angular.z
        
        return vel
    
    def get_tune_parameters(self):
        cg, cg_inflation = self.get_parameters()
        params = []
        for itr in range(len(self.list_tune_params)):
            name = self.list_tune_params[itr][0]
            one_param = []
            one_param.append(name)
            if self.list_tune_params[itr][0] == 'inflation_radius':
                one_param.append(cg_inflation[name])
            else:
                if name == 'vel':
                    one_param.append(cg['max_vel_x'])
                else:
                    one_param.append(cg[name])
            params.append(one_param)
        return params

    # Local goal related functions

    def get_robot_position(self):
        odom = self.get_odom()
        X = odom.pose.pose.position.x
        Y = odom.pose.pose.position.y
        return X, Y

    def get_robot_status(self):
        odom = self.get_odom()
        q1 = odom.pose.pose.orientation.x
        q2 = odom.pose.pose.orientation.y
        q3 = odom.pose.pose.orientation.z
        q0 = odom.pose.pose.orientation.w
        X = odom.pose.pose.position.x
        Y = odom.pose.pose.position.y
        Z = odom.pose.pose.position.z
        PSI = np.arctan2(2 * (q0*q3 + q1*q2), (1 - 2*(q2**2+q3**2)))
        qt = (q1, q2, q3, q0)
        return X, Y, Z, PSI, qt

    def transform_gp(self, gp, X, Y, PSI):

        R_r2i = np.matrix([[np.cos(PSI), -np.sin(PSI), X], [np.sin(PSI), np.cos(PSI), Y], [0, 0, 1]])
        R_i2r = np.linalg.inv(R_r2i)

        pi = np.concatenate([gp, np.ones_like(gp[:, :1])], axis=-1)
        pr = np.matmul(R_i2r, pi.T)
        return np.asarray(pr[:2, :])

    def transform_lg(self, wp, X, Y, PSI):
        """
        Change orientation of point in path according to egocentric orientation.
        """
            
        R_r2i = np.matrix([[np.cos(PSI), -np.sin(PSI), X], [np.sin(PSI), np.cos(PSI), Y], [0, 0, 1]])
        R_i2r = np.linalg.inv(R_r2i)
        pi = np.matrix([[wp[0]], [wp[1]], [1]])
        pr = np.matmul(R_i2r, pi)
        lg = np.array([pr[0,0], pr[1, 0]])
        return lg

    def get_global_path_and_robot_status(self):
        gp = self.get_gp()
        #print(f"gp: {gp}")

        ### If no path is found
        if (isinstance(gp, bool) and gp is False) or (isinstance(gp, int)) :
            return False, False, False, False
        
        X, Y, Z, PSI, qt = self.get_robot_status()

        gp = self.transform_gp(gp, X, Y, PSI)

        return gp.T, X, Y, PSI

    def get_local_goal_angle(self, los = 1.75):
        """Get the local goal coordinate relative to the robot's current location

        Returns:
            [Pose msg]: pose msg with attributes x, y, and orientaton
        """

        gp, X, Y, PSI = self.get_global_path_and_robot_status()
        if isinstance(gp, bool) and gp is False:
            return 0, 0, False, 0, 0
        
        #los = 1.75 # Before 1.75
        egocentric = True

        lg_x = 0
        lg_y = 0
        lg = np.zeros(2)
        itr = 0
        if len(gp)>0:
            lg_flag = 0
            for wp in gp:
                itr += 1
                dist = (np.array(wp)-np.array([0, 0]))**2 # Before X and Y
                #print(f"X: {X}, wp[0]: {wp[0]}")
                #print(f"Y: {Y}, wp[1]: {wp[1]}")
                dist = np.sum(dist, axis=0)
                #print(dist)
                dist = np.sqrt(dist)
                if dist > los:
                    #print(f"dist is : {dist}, itr is :{itr}")
                    lg_flag = 1
                    if egocentric:
                        #lg = self.transform_lg(wp, 0, 0, 0) 
                        lg = wp
                    else:
                        lg = self.transform_lg(wp, X, Y, PSI)
                    lg_x = lg[0]
                    lg_y = lg[1]
                    break
            if lg_flag == 0:
                if egocentric:
                    #lg = self.transform_lg(gp[-1], 0, 0, 0)
                    lg = gp[-1]
                else:
                    lg = self.transform_lg(gp[-1], X, Y, PSI)
                lg_x = lg[0]
                lg_y = lg[1]

        local_goal = np.array([np.arctan2(lg_y, lg_x)]) # Before lg_y, lg_x


        # Calculate the coordinates of the point at the specified distance and angle
        # x_new = gp[0, 0] + 2 * np.cos(local_goal)
        # y_new = gp[0, 1] + 2 * np.sin(local_goal)

        # x_ego = gp[0, 0] + 2 * np.cos(PSI)
        # y_ego = gp[0, 1] + 2 * np.sin(PSI)

        # self.counter_gp += 1

        # if self.counter_gp == 400:

        #     self.counter_gp = 0 

        #     plt.scatter(gp[:itr, 0], gp[:itr, 1], color='red', label='First ' + str(itr) + ' points')
        #     plt.scatter(gp[itr:, 0], gp[itr:, 1], color='blue', label='Remaining points')

        #     # Connect the marked point to the first point
        #     plt.scatter(x_new, y_new, color='yellow', label='Marked Point')
        #     plt.plot([gp[0, 0], x_new], [gp[0, 1], y_new], color='gray', linestyle='--')

        #     # Connect the marked point to the first point
        #     plt.scatter(x_ego, y_ego, color='green', label='Marked Point')
        #     plt.plot([gp[0, 0], x_ego], [gp[0, 1], y_ego], color='gray', linestyle='--')

        #     plt.xlabel('X Position')
        #     plt.ylabel('Y Position')
        #     plt.title('Scatter Plot of X and Y Positions')
        #     plt.gca().set_aspect('equal')
        #     plt.grid(True)
        #     plt.show(block=False)
        #     plt.pause(4)
        #     plt.close()

        return local_goal, PSI, gp, itr, np.array(lg)

    def get_dist_and_wp(self):
       """Get current odom and waypoint distance + waypoint in global frame.
        """
       los = 1.75 # Before 1.75 
       
       # Try to get gp at most three times
       for i in range(3):
           gp = self.get_gp()
           if (isinstance(gp, bool) and gp is False) or (isinstance(gp, int)) :
                gp = self.get_gp()
           else:
               break
       if (isinstance(gp, bool) and gp is False) or (isinstance(gp, int)):
           return False, False
       
       X, Y = self.get_robot_position()
       itr = 0
       
       wpp = np.zeros(2)
       for wp in gp:
            itr += 1
            wpp = wp
            dist = (np.array(wp)-np.array([X, Y]))**2 # Before X and Y
            #print(f"X: {X}, wp[0]: {wp[0]}")
            #print(f"Y: {Y}, wp[1]: {wp[1]}")
            dist = np.sum(dist, axis=0)
            dist = np.sqrt(dist)
            #print(dist)
            if dist > los:
                #print("!!!!!!!!!HERE!!!!!!!!!!")
                #print(f"iter: {itr}: xy: {np.array([X, Y])} wp: {wp}")
                break
        
       return np.array([X, Y]), np.array(wpp)
    
    def get_dist(self, a, b):
        """ Calculate distance between two points
        """
        dist = (a-b)**2
        dist = np.sum(dist, axis=0)
        dist = np.sqrt(dist)

        return dist

    def get_dist_traveled(self, init_point, wp):
        """ Calculate distance traveled
        """

        X, Y = self.get_robot_position()
        final_point = np.array([X, Y])

        init_dist = self.get_dist(init_point, wp)
        #print(f"inital distance: {init_dist}")
        final_dist = self.get_dist(final_point, wp)
        #print(f"final distance: {final_dist}")

        traveled_dist = init_dist - final_dist
        #print(f"{traveled_dist}")

        return traveled_dist



    
    def limit_sum(self, num1, num2):
        result = round(num1 + num2)
        
        if result < 0:
            result = 0
        elif result > 59:
            result = 59
            
        return result

    def add_angle_to_costmap(self, costmap_np, angle, PSI):
        #costmap_np *= 0.9
        angle *= 30

        # Make angle a square point in te costmap
        #print(f"PSI: {PSI}")
        rotated_costmap = rotate(costmap_np, np.degrees(PSI), reshape=False)
        rotated_costmap_and_angle = rotated_costmap*0.8 # rotated_costmap*0.8

        # for itr in range(2):
        #     for jtr in range(2):
        #         rotated_costmap_and_angle[self.limit_sum((31 - itr if angle[0] < 0 else 28 + itr), angle[0]), 
        #                                   self.limit_sum((31 - jtr if angle[1] < 0 else 28 + jtr), angle[1])] = 1

        for itr in range(3):
            rotated_costmap_and_angle[self.limit_sum((32 - itr if angle[0] < 0 else 27 + itr), angle[0]), 
                                          self.limit_sum((31 if angle[1] < 0 else 28 ), angle[1])] = 1
        for itr in range(3):
            rotated_costmap_and_angle[self.limit_sum((31 if angle[0] < 0 else 28 ), angle[0]), 
                                        self.limit_sum((32 - itr if angle[1] < 0 else 27 + itr), angle[1])] = 1

        return rotated_costmap, rotated_costmap_and_angle

    def add_action_to_costmap(self, costmap_np_angle, action):
        #costmap_np *= 0.9

        x = 30 + round(action[1]*15)
        y = 30 + round(action[0]*15)

        x_0 = max(0 + 3, min(x, 60 - 3))
        x_1 = max(0 + 3, min(x + 2 if x > 0 else x - 2, 60 - 3))

        y_0 = max(0 + 3, min(y, 60 - 3))
        y_1 = max(0 + 3, min(y + 2 if y > 0 else y - 2, 60 - 3))

        costmap_np_angle_and_action = costmap_np_angle
        costmap_np_angle_and_action[x_0:x_1, y_0:y_1] = 1  # Draw the vertical line of the "T"
        #costmap_np_angle_and_action[y_h, x1_h:x2_h+1] = 1  # Draw the horizontal line of the "T"

        return costmap_np_angle_and_action
    
    def pure_pursuit(self):
        max_tries = 5
        los = 0.3
        tries, angle_atan, PSI, gp, gp_len, wp = self.try_get_path(max_tries, los) # Get waypoint angle
        if tries == max_tries: # If path is not found
            return np.zeros(2)
        angle = np.arctan2(wp[0], wp[1])
        linear_x = 0.5*np.sign(np.sin(angle))
        #angular_z = 0.5*np.sign(np.cos(angle))
        angular_z = np.cos(angle)

        return np.array([linear_x, angular_z])


    def supervisor(self):
        """ Check if robot is not safe.
            Currently implemented with proximity.
        """
        lidar = self.get_filtered_scan()
        min_range = np.min(lidar)

        if min_range < 0.2:
            #print("here")
            return False
        else:
            return False

    def generate_sample(self, a, b, mean, std_dev):
        sample = truncnorm.rvs(a, b, loc=mean, scale=std_dev, size=1)
        return sample[0]


    def randomize_action(self, action):
        """ Add noise to dwa_action. Keep it for a second.
        """
        random_action = np.zeros(2)
        self.random_counter += 1 
        if self.random_counter == 5:
            self.random_counter = 0
            
            # Normal distribution
            #self.random_noise_vel = max(-0.25, min(np.random.normal(0, 0.15), 0.25))
            #self.random_noise_ang = max(-0.15, min(np.random.normal(0, 0.1), 0.15))
            # Uniform
            self.random_noise_vel = random.uniform(-0.15, 0.5) ### random.uniform(-0.15, 0.5)
            self.random_noise_ang = random.uniform(-0.25, 0.25) ### random.uniform(-0.25, 0.25)
            # Normal
            #self.random_noise_vel = self.generate_sample((-0.15 - 0.3) / 0.1, (0.45 - 0.3) / 0.1, 0.3, 0.1)
            #self.random_noise_ang = self.generate_sample((-0.5 - 0) / 0.1, (0.5 - 0) / 0.1, 0, 0.1)
            if action[0] < 0:
                self.random_noise_vel = -self.random_noise_vel
        random_action[0] = action[0] + self.random_noise_vel
        random_action[1] = action[1] + self.random_noise_ang
        #random_action[1] = action[1] + 0#self.random_noise_ang
        #print(f"action: {action}")
        #print(f"random: {random_action}")

        return random_action

    def random_action(self):
        """ Random noise. Keep it for a second.
        """
        random_action = np.zeros(2)
        self.random_counter += 1 
        if self.random_counter == 5:
            self.random_counter = 0
            
            self.random_noise_vel = random.uniform(-1.5, 1.5) 
            self.random_noise_ang = random.uniform(-1.5, 1.5) 
        random_action[0] = self.random_noise_vel
        random_action[1] = self.random_noise_ang

        return random_action

    def modify_vel(self, dwa_vel, control):
        """ Changes velocity given by dwa_vel according to control
        """
        print(f"original dwa: {dwa_vel}")
        new_vel = copy.deepcopy(dwa_vel)
        if control is not None:
            if control == 0:
                new_vel[0] += 0.25 if new_vel[0] > 0 else -0.25
                new_vel[1] += 0.45
            elif control == 1:
                new_vel[0] += -0.25
            elif control == 2:
                new_vel[0] += 0.25 if new_vel[0] > 0 else -0.25
                new_vel[1] += -0.45
            elif control == 3:
                new_vel[0] += 0.25
        print(f"Modified dwa: {new_vel}")

        return new_vel
            
    
    def is_robot_stuck(self):
        # If the robot absoulte velocities sum to a number smaller than 1 in the last 100 steps (10 seconds)
        # the robot is stuck
        odom  = self.get_odom_buffer()
        s = sum(odom)
        info = None
        if s < 1:
            self.stuck += 1
        else:
            self.stuck = 0

        if self.stuck >= 50:
            info = "Environment crashed"

        return 
    
    def reward_applr(self, obs):
        # Robot velocity
        odom = self.get_odom()
        vel_rob = np.abs(odom.twist.twist.linear.x)

        # Minimum distance
        min_range = np.min(obs)

        cf = -1
        cp = 1
        cc = -1

        reward_func = cf*self.max_vel_robot + cp*vel_rob + cc*1/min_range # For RL: velocity and obstacles equally

        return reward_func

    
    def reward_func(self, obs):
        # Robot velocity
        odom = self.get_odom()
        vel_rob = np.abs(odom.twist.twist.linear.x)

        # Minimum distance
        min_range = np.min(obs)

        #print(vel_rob)
        #print(min_range)

        ###########################################################################################################
        ##### Reward function Bayesian - Immediate reward: velocity of robot + distance + relative velocity #######
        ###########################################################################################################

        reward_func = vel_rob*(-1 if min_range < 0.75 else 1) - self.max_vel_robot # For RL: velocity and obstacles equally
        #reward_func = (np.tanh((vel_rob-1)*5)+1)*(-100 if min_range < 0.75 else 0) - 0.1 # For RL: penalize greatly high velocities near obstacles
        #reward_func = vel_rob*(-5 if min_range < 0.75 else 1) - self.max_vel_robot # For RL: velocity and obstacles. Obstacle smore important than velocity
        #print(f"min_range: {min_range}, vel_rob: {vel_rob}, reward_func: {reward_func}")
        #reward_func = vel_rob*(-1 if min_range < 0.75 else 1)

        #reward_func = 4*vel_rob*(-1 if min_range < 0.8 else 1) - 0.1*(-1 if min_range < 0.5 else 0)
        #reward_func = 4*vel_rob*np.tanh((min_range-0.4)/0.2) - 0.1*(1/min_range)
        #reward_func = 4*vel_rob*np.tanh((min_range-0.8))
        #print(reward_func)
        #reward_func = - 0.1*(1/min_range)
        #print(reward_func)

        ###########################################################################################################
        #### Reward function DRL - Accumulated reward: velocity of robot when close to obstacles + distance - 1 ###
        ###########################################################################################################

        #reward_func = vel_rob*np.tanh((min_range-0.8)*4 if min_range < 0.8 else 0) - 0.1*(1/min_range) - 1
        #reward_func = vel_rob*np.tanh((min_range-0.8)*4 if min_range < 0.8 else 0) 
        #reward_func = - 0.1*(1/min_range)
        #reward_func = - 1

        return reward_func

    def reward_feedback(self, state, action, rl_distribution):

        sft_distribution = self.sft.act(torch.FloatTensor(state).to(self.device)).data.cpu().numpy()
        #print(sft_distribution)

        kl = np.sum(np.where(sft_distribution != 0, sft_distribution * np.log(sft_distribution / rl_distribution.data.cpu().numpy()), 0))

        kl = np.clip(kl, 0, 10)


        r = -self.reward_net.act(state, action) - kl


        r = float(r)

        return r

    def reward_function_cdrl(self, costmap_np_angle, action, dwa_action):
        # Reward function of CDRL: kl(move_base, current_action) + human_reward

        costmap_np_angle_and_action = self.add_action_to_costmap(costmap_np_angle, action)

        # Human reward function
        thr = 0.5
        human_reward = self.reward_cdrl.act(torch.FloatTensor(costmap_np_angle_and_action).to(self.device)).data.cpu().numpy() >= thr

        # Similarity reward
        vel_divergence = np.sqrt( (action[0] - dwa_action[0])**2 + (action[1] - dwa_action[1])**2 )

        reward = -( vel_divergence + human_reward )

        #return -vel_divergence
        #print(f"reward vel: {vel_divergence}")

        return reward[0][0]

    def reward_function_cdrl_short(self, traveled_dist, linear_x):
        """ reward = -0.25 + 5*traveled_dist + collision
        """
        collision = 0
        lidar = self.get_filtered_scan()
        min_range = np.min(lidar)
        linear_x = np.abs(linear_x)
        #print(f"{min_range}")
        if min_range < 0.5: # Before 0.2 Before 0.5
            collision = -4*linear_x - 1 
            #collision = -1

        #reward = -0.5 + 5*traveled_dist + collision
        #reward = -0.5 + 5*traveled_dist + collision # REWARD NO WAYPOINT
        #reward = -0.25 + 5*traveled_dist + collision ### reward = -0.5 + 5*traveled_dist + collision
        reward = -0.5 + collision # REWARD WAYPOINT
        #reward = collision
        return reward

    def supervisor_short(self):
        """ Check if robot is not safe.
            FUnction of proximity and velocity angle.
        """     
        X, Y, Z, PSI, qt = self.get_robot_status()
        odom = self.get_odom()
        linear_x = odom.twist.twist.linear.x
        linear_x_abs = np.abs(linear_x)
        lidar = self.get_filtered_scan()
        len_lidar = 720
        safe_range = 30 # 15 degrees
        left_range = 0
        right_range = 0


        if linear_x_abs < 0:
            PSI = PSI + np.pi

        if PSI > 0:
            lidar_angle = (PSI/(np.pi*2))*len_lidar
        if PSI < 0:
            lidar_angle = len_lidar + (PSI/(np.pi*2))*len_lidar

        min_index = np.argmin(lidar)
        print(f"min_index:{min_index}")
        #print(f"PSI:{PSI}")
        print(f"lidar_angle: {lidar_angle}")

        left_range = (min_index + safe_range) % len_lidar
        right_range = (min_index - safe_range) % len_lidar

        if left_range > safe_range:
            if (lidar_angle < left_range) and (lidar_angle > right_range):
                print("SUPERVISOR")
                return True
        else:
            print("DISCONTINUITY")
            if (lidar_angle < left_range) or (lidar_angle > right_range):
                print("SUPERVISOR")
                return True

        return False

    def supervisor_short_v2(self):
        """ Check if robot is not safe.
            FUnction of proximity and velocity angle.
        """     
        odom = self.get_odom()
        linear_x = odom.twist.twist.linear.x
        lidar = self.get_filtered_scan()
        len_lidar = 720
        safe_range = 90 # 15 degrees
        left_range = 0
        right_range = 0

        min_range = np.min(lidar)
        min_index = np.argmin(lidar)

        if min_range < 0.5:
            if (min_index < safe_range) or (min_index > (len_lidar - safe_range)):
                if linear_x > 0:
                    print("SUPERVISOR front")
                    return 1
            if (min_index < (len_lidar/2  + safe_range)) and (min_index > (len_lidar/2  - safe_range)):
                if linear_x < -0:
                    print("SUPERVISOR back")
                    return 1
                
        if min_range < 0.2:
            #print(f"min_range: {min_range}")
            print("SUPERVISOR general")
            return 1
        
        if self.supervisor_counter == 0:
            return 0
        
        self.supervisor_counter += 1
        self.supervisor_counter = self.supervisor_counter % self.max_supervisor

        return self.supervisor_counter

        return 0

    def supervisor_fuzzy(self, x):
        """ Check if robot is not safe.
            FUnction of proximity and velocity angle.
        """     
        v_lo = x[0]
        v_hi = x[1]
        a_lo = x[2]
        a_hi = x[3]

        odom = self.get_odom()
        linear_x = odom.twist.twist.linear.x
        angular_z = odom.twist.twist.angular.z
        lidar = self.get_filtered_scan()
        len_lidar = 720
        safe_range = 45

        min_range = np.min(lidar)
        min_index = np.argmin(lidar)

        # Activate supervisor 
        if min_range < 1:

            if min_range < 0.2:
                #print(f"min_range: {min_range}")
                print("SUPERVISOR general")
                return 1, True
            
            dist_defuzzy, ang_defuzzy = self.fuzzy_compute(v_lo, v_hi, a_lo, a_hi, np.abs(linear_x), np.abs(angular_z))

            # If distance to object is less than dist_defuzzy
            if min_range < dist_defuzzy:
                if linear_x > 0:
                    if angular_z > 0:
                        if (min_index < safe_range) or (min_index > (len_lidar - ang_defuzzy)):
                            print("SUPERVISOR front right")
                            return 1, False
                    if angular_z < 0:
                        if (min_index < ang_defuzzy) or (min_index > (len_lidar - safe_range)):
                            print("SUPERVISOR front left")
                            return 1, False
                if linear_x < 0:
                    if angular_z > 0:
                        if (min_index < (len_lidar/2  + safe_range)) and (min_index > (len_lidar/2  - ang_defuzzy)):
                            print("SUPERVISOR back right")
                            return 1, False
                    if angular_z < 0:
                        if (min_index < (len_lidar/2  + ang_defuzzy)) and (min_index > (len_lidar/2  - safe_range)):
                            print("SUPERVISOR back left")
                            return 1, False
        return 0, False

    def mini_supervisor(self):
        """ return true if min_range < 0.15
        """
        lidar = self.get_filtered_scan()
        min_range = np.min(lidar)
        min_index = np.argmin(lidar)
        len_lidar = 720
        if min_range < 0.15:
            print("MINI SUPERVISOR")
            if (min_index < len_lidar/4) or (min_index > (len_lidar*3)/4):
                return True, True
            else:
                return True, False
        return False, False


    def supervisor_fuzzy_v2(self, x):
        """ Check if robot is not safe.
            FUnction of proximity and velocity angle.
        """     
        v_lo = x[0]
        v_hi = x[1]
        a_lo = x[2]
        a_hi = x[3]
        #a_lo = 0.5
        #a_hi = 0.5

        odom = self.get_odom()
        linear_x = odom.twist.twist.linear.x
        angular_z = odom.twist.twist.angular.z
        lidar = self.get_filtered_scan()
        len_lidar = 720
        safe_range = 60

        min_range = np.min(lidar)
        min_index = np.argmin(lidar)

        # Activate supervisor 
        if min_range < 1:

            #print(f"{min_range}, {min_index}")

            if min_range < 0.2:
                #print(f"min_range: {min_range}")
                print("SUPERVISOR general")
                if (min_index < len_lidar/4) or (min_index > (len_lidar*3)/4):
                    #print(f"Front: {min_index}")
                    return 1, True, True
                else:
                    #print(f"Back: {min_index}")
                    return 1, True, False
            
            dist_defuzzy, ang_defuzzy = self.fuzzy_compute(v_lo, v_hi, a_lo, a_hi, np.abs(linear_x), np.abs(angular_z))

            safe_range = ang_defuzzy

            # If distance to object is less than dist_defuzzy
            if min_range < dist_defuzzy:
                if linear_x > 0:
                    if (min_index < safe_range) or (min_index > (len_lidar - safe_range)):
                        print("SUPERVISOR front right")
                        return 1, False, None
                if linear_x < 0:
                    if (min_index < (len_lidar/2  + safe_range)) and (min_index > (len_lidar/2  - safe_range)):
                        print("SUPERVISOR back right")
                        return 1, False, None
        return 0, False, None

    def supervisor_fuzzy_v3(self, x):
        """ Check if robot is not safe.
            FUnction of proximity and velocity angle.
        """     
        v_lo = x[0]
        v_hi = x[1]
        a_lo = x[2]
        a_hi = x[3]
        #a_lo = 0.5
        #a_hi = 0.5

        odom = self.get_odom()
        linear_x = odom.twist.twist.linear.x
        angular_z = odom.twist.twist.angular.z
        lidar = self.get_filtered_scan()
        len_lidar = 720
        safe_range = 60

        min_range = np.min(lidar)
        min_index = np.argmin(lidar)

        # Activate supervisor 
        if min_range < 1:

            #print(f"{min_range}, {min_index}")
            #print(f"min_range: {min_range}")
            if min_range < 0.3:
                #print(f"min_range: {min_range}")
                print("SUPERVISOR general")
                if (min_index < len_lidar/4) or (min_index > (len_lidar*3)/4):
                    #print(f"Front: {min_index}")
                    return 1, True, True
                else:
                    #print(f"Back: {min_index}")
                    return 1, True, False
            
            dist_defuzzy, ang_defuzzy = self.fuzzy_compute(v_lo, v_hi, a_lo, a_hi, np.abs(linear_x), np.abs(angular_z))

            # If distance to object is less than dist_defuzzy
            if min_range < dist_defuzzy:
                print(f"dist_defuzzy:{dist_defuzzy}")
                print("SUPERVISOR fuzzy")
                return 1, False, None
            
        return 0, False, None

    def fuzzy_membership(self, v_lo, v_hi, a_lo, a_hi):

        n = 100

        # Linear velocity
        v = np.linspace(0, 1.5, n)
        v_lo_mf = fuzz.gaussmf(v, 0, v_lo)
        v_hi_mf = fuzz.gaussmf(v, 1.5, v_hi)

        # Angular velocity
        a = np.linspace(0, 1.5, n)
        a_lo_mf = fuzz.gaussmf(a, 0, a_lo)
        a_hi_mf = fuzz.gaussmf(a, 1.5, a_hi)

        # Distance
        # dist = np.linspace(0, 1.3, n)
        # dist_lo_mf = fuzz.gaussmf(dist, 0, 0.3)
        # dist_hi_mf = fuzz.gaussmf(dist, 1.3, 0.3)
        dist = np.linspace(-0.5, 2, n)
        dist_lo_mf = fuzz.gaussmf(dist, -0.5, 0.5)
        dist_hi_mf = fuzz.gaussmf(dist, 2, 0.5)

        # Angle
        ang = np.linspace(0, 225, n)
        ang_lo_mf = fuzz.gaussmf(ang, 0, 60)
        ang_hi_mf = fuzz.gaussmf(ang, 225, 60)

        return v_lo_mf, v_hi_mf, a_lo_mf, a_hi_mf, dist_lo_mf, dist_hi_mf, ang_lo_mf, ang_hi_mf

    def fuzzy_fuzzify(self, v_lo_mf, v_hi_mf, a_lo_mf, a_hi_mf, vel, ang):

        n = 100
        v = np.linspace(0, 1.5, n)
        a = np.linspace(0, 1.5, n)

        # Linear velocity
        v_level_lo = fuzz.interp_membership(v, v_lo_mf, vel)
        v_level_hi = fuzz.interp_membership(v, v_hi_mf, vel)

        # Angular velocity
        a_level_lo = fuzz.interp_membership(a, a_lo_mf, ang)
        a_level_hi = fuzz.interp_membership(a, a_hi_mf, ang)

        return v_level_lo, v_level_hi, a_level_lo, a_level_hi


    def fuzzy_rules(self, v_level_lo, v_level_hi, a_level_lo, a_level_hi, dist_lo_mf, dist_hi_mf, ang_lo_mf, ang_hi_mf):

        # If linear velocity is low THEN distance is low
        dist_activation_lo = np.fmin(v_level_lo, dist_lo_mf)
        # If linear velocity is high THEN distance is high
        dist_activation_hi = np.fmin(v_level_hi, dist_hi_mf)
        # Distance Aggregation
        dist_aggregated = np.fmax(dist_activation_lo, dist_activation_hi)

        # If linear velocity is low AND angular velocity is low THEN angle is low
        ang_activation_lo = np.fmin(v_level_lo, a_level_lo)
        ang_activation_lo = np.fmin(ang_activation_lo, ang_lo_mf)
        # If linear velocity is high OR angular velocity is high THEN angle is high
        ang_activation_hi = np.fmax(v_level_hi, a_level_hi)
        ang_activation_hi = np.fmin(ang_activation_hi, ang_hi_mf)
        # Angle Aggregation
        ang_aggregated = np.fmax(ang_activation_lo, ang_activation_hi)

        #print(f"dist_aggregated: {dist_aggregated}, ang_aggregated: {ang_aggregated}")

        return dist_aggregated, ang_aggregated

    def fuzzy_rules_v3(self, v_level_lo, v_level_hi, a_level_lo, a_level_hi, dist_lo_mf, dist_hi_mf, ang_lo_mf, ang_hi_mf):

        # If linear velocity is low AND angular velocity is low THEN distance is low
        dist_activation_lo = np.fmin(v_level_lo, a_level_lo)
        dist_activation_lo = np.fmin(dist_activation_lo, dist_lo_mf)
        # If linear velocity is high OR angular velocity is high THEN distance is high
        dist_activation_hi = np.fmax(v_level_hi, a_level_hi)
        dist_activation_hi = np.fmin(dist_activation_hi, dist_hi_mf)
        # Distance Aggregation
        dist_aggregated = np.fmax(dist_activation_lo, dist_activation_hi)

        # If linear velocity is low AND angular velocity is low THEN angle is low
        ang_activation_lo = np.fmin(v_level_lo, a_level_lo)
        ang_activation_lo = np.fmin(ang_activation_lo, ang_lo_mf)
        # If linear velocity is high OR angular velocity is high THEN angle is high
        ang_activation_hi = np.fmax(v_level_hi, a_level_hi)
        ang_activation_hi = np.fmin(ang_activation_hi, ang_hi_mf)
        # Angle Aggregation
        ang_aggregated = np.fmax(ang_activation_lo, ang_activation_hi)

        #print(f"dist_aggregated: {dist_aggregated}, ang_aggregated: {ang_aggregated}")

        return dist_aggregated, ang_aggregated


    def fuzzy_defuzzyfication(self, dist_aggregated, ang_aggregated):

        n = 100
        # Defuzzify distance
        if np.max(dist_aggregated) == 0:
            dist_defuzzy = 0.3
        else:
            #dist = np.linspace(0, 1.3, n)
            dist = np.linspace(-0.5, 2, n)
            dist_defuzzy = fuzz.defuzz(dist, dist_aggregated, 'centroid')

        # Defuzzify angle
        if np.max(ang_aggregated) == 0:
            ang_defuzzy = 45
        else:
            ang = np.linspace(0, 225, n)
            ang_defuzzy = fuzz.defuzz(ang, ang_aggregated, 'centroid')
        
        
        clipped_dist_defuzzy= np.clip(dist_defuzzy, 0.25, 1)
        clipped_ang_defuzzy= np.clip(ang_defuzzy, 45, 180)

        #print(f"clipped_dist_defuzzy:{clipped_dist_defuzzy}, {clipped_ang_defuzzy}")

        return clipped_dist_defuzzy, clipped_ang_defuzzy

    def fuzzy_compute(self, v_lo, v_hi, a_lo, a_hi, vel, ang):

        # Obtain membership functions
        v_lo_mf, v_hi_mf, a_lo_mf, a_hi_mf, dist_lo_mf, dist_hi_mf, ang_lo_mf, ang_hi_mf = self.fuzzy_membership(v_lo, v_hi, a_lo, a_hi)

        # x = np.linspace(0, 1.5, 100)
        # y = np.linspace(0, 1.3, 100)
        # z = np.linspace(0, 225, 100)

        # plt.plot(z, ang_lo_mf)
        # plt.plot(z, ang_hi_mf)

        # plt.show(block=False)
        # plt.pause(5)
        # plt.close()

        # Obtain values (Fuzzify inputs)
        v_level_lo, v_level_hi, a_level_lo, a_level_hi = self.fuzzy_fuzzify(v_lo_mf, v_hi_mf, a_lo_mf, a_hi_mf, vel, ang)

        #print(f"vel: {vel}, ang: {ang}")
        #print(f"v_level_lo: {v_level_lo},v_level_hi: {v_level_hi}, a_level_lo:{a_level_lo} , a_level_hi:{a_level_hi}")

        # Rule operation (Inference and aggregation)
        dist_aggregated, ang_aggregated = self.fuzzy_rules_v3(v_level_lo, v_level_hi, a_level_lo, a_level_hi, dist_lo_mf, dist_hi_mf, ang_lo_mf, ang_hi_mf)

        # Obtain crisp values (Defuzzify)
        dist_defuzzy, ang_defuzzy = self.fuzzy_defuzzyfication(dist_aggregated, ang_aggregated)

        return dist_defuzzy, ang_defuzzy
    



