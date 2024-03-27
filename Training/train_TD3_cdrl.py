#!/usr/bin/env python3

import numpy as np
import torch
import argparse
import os

import utils_TD3_cdrl_try
import TD3_cdrl_try
import rospy
import task_env
import os
from os import listdir
from os.path import isfile, join
import yaml
from yaml.loader import SafeLoader
import time
import ruamel.yaml
import datetime
import wandb

from PIL import Image
import matplotlib.pyplot as plt


USE_WANDB = False

# Initialize WandB
if USE_WANDB:
	wandb.init(project="SDRL", name="RL DPO| Waypoint", config={"hyperparameter": {"CNN": "V3", "no_supevisor": True}})


dir_path = os.path.dirname(os.path.realpath(__file__))

# Curriculum learning: Number of episodes | max_episodes
curriculum = np.zeros([2,2])

curriculum[0,0] = 6e3
curriculum[0,1] = 400 #400

curriculum[1,0] = 200e3
curriculum[1,1] = 400 #400


curriculum_counter = 0
curriculum_goals = 2
max_time_steps = 400 #400

#### For recording ####
buffer_len = 16
ptr = 0
record_counter = 0
vel_buffer = np.zeros([buffer_len, 2])
goal_buffer = np.zeros([buffer_len, 2])
state_buffer = np.zeros([buffer_len, 120, 120])
joystick_buffer = np.zeros([buffer_len, 2])
########################



def switch_curriculum(env):
	global curriculum_counter, curriculum, curriculum_goals, max_time_steps
	curriculum_counter += 1
	curriculum_counter %= curriculum_goals

	env.switch_goal(curriculum_counter)
	max_time_steps = curriculum[curriculum_counter, 1]


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env, eval_episodes=6):
	global max_time_steps
	print("---------------------------------------")
	print("Starting evaluation")
	print("---------------------------------------")
	avg_reward = 0.
	for _ in range(eval_episodes):
		vel, goal, state, done = env.reset()
		counter = 0
		while not done:
			counter += 1
			action = policy.select_action(state, vel, goal)
			vel, goal, state, reward, done = env.step_cdrl_short_wp(action)
			avg_reward += reward

			if counter > max_time_steps:
				done = True


	avg_reward /= eval_episodes

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	return avg_reward


def main():

	global curriculum, curriculum_counter, max_time_steps 
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG or OurDDPG)
	parser.add_argument("--env", default="PTDRL")          # OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--start_timesteps", default=100*8e3, type=int)# Time steps initial random policy is used 25e3 ### 50e3, 2e3, 8e3
	parser.add_argument("--eval_freq", default=8e3, type=int)       # How often (time steps) we evaluate - Before 5e3, now 3e4 ### 25e3, 2e3
	parser.add_argument("--max_timesteps", default=6e4, type=int)   # Max time steps to run environment. Between 15.000 to 150.000 episodes #1e7
	parser.add_argument("--max_time_per_episode", default=2000, type=int)  # Max episodes to run environment
	parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise ### Before 0.1
	parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic - Before 256
	parser.add_argument("--discount", default=0.9)                 # Discount factor - before 0.99 -- 0.9
	parser.add_argument("--tau", default=0.005)                     # Target network update rate -- 0.005
	parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update ### Before 0.2
	parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise ### Before 0.5
	parser.add_argument("--policy_freq", default=4, type=int)       # Frequency of delayed policy updates ### Before 2 ############## CDRL 4 #######
	parser.add_argument("--save_model", default = False, action="store_true")        # Save model and optimizer parameters
	parser.add_argument("--load_model", default=True)                 # Model load file name, "" doesn't load, "default" uses file_name
	args, unknown = parser.parse_known_args()
	

	file_name = f"{args.policy}_{args.env}_{args.seed}"
	print("---------------------------------------")
	print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")

	if not os.path.exists(dir_path + "/results/cdrl"):
		os.makedirs(dir_path + "/results/cdrl")

	if args.save_model and not os.path.exists(dir_path + "/models_TD3_cdrl"):
		os.makedirs(dir_path + "/models_TD3_cdrl")

	env = task_env.PtdrlTaskEnv()
	episode_num, min_eval, time_step, ptr, size = get_env_params()
	print(f"episode_num: {episode_num}")

	# Set seeds
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	
	#state_dim = 60 # Latent space 10. 10x4=40
	state_dim = 120 # Latent space 10. 10x4=40
	action_dim = 2 # 8 parameters to tune
	max_action = 1.5 # Maybe separate linear and angular vel

	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
	}

	# Initialize policy
	if args.policy == "TD3":
		# Target policy smoothing is scaled wrt the action scale
		kwargs["policy_noise"] = args.policy_noise * max_action
		kwargs["noise_clip"] = args.noise_clip * max_action
		kwargs["policy_freq"] = args.policy_freq
		policy = TD3_cdrl_try.TD3(**kwargs)
	
	#####
	initial_params = policy.actor.state_dict()
	print("Initial Parameters:")
	#print(initial_params)
	#####

	if args.load_model:
		policy_file = file_name
		#policy.load(f"{dir_path}/models_TD3_cdrl/{policy_file}", "current")
		#policy.load(dir_path + "/models_TD3_cdrl/22/models/TD3_PTDRL_0", "current")
		policy.load_dpo(dir_path + "/models_TD3_cdrl/22/models/TD3_PTDRL_0", dir_path + "/DPO/TD3_PTDRL_22_actor_current_dpo_simul2")
		print("Loading Network!!!")

	replay_buffer = utils_TD3_cdrl_try.ReplayBufferGoal(state_dim, action_dim)
	
	# Evaluate untrained policy
	evaluations = np.zeros([0])
	all_rewards = np.zeros([0])
	critic_losses = np.zeros([0])
	actor_losses = np.zeros([0])

	if episode_num > 0 :
		all_rewards, critic_losses, actor_losses = load_results()
		replay_buffer.load(dir_path)
		replay_buffer.ptr = ptr
		replay_buffer.size = size
		evaluations = np.load(f"{dir_path}/results/cdrl/{file_name}.npy")

	# Start training
	#state, done = env.reset(), False
	#goal, state, done = env.reset()
	vel, goal, state, done = env.reset()
	#time.sleep(1)
	

	episode_reward = 0
	episode_timesteps = 0
	#switch_curriculum(env)
	#max_time_steps = 400#50#100

	for t in range(time_step, int(args.max_timesteps)):
		episode_timesteps += 1

		# Update goals and max_time_steps
		#if t == curriculum[curriculum_counter, 0]:
			#switch_curriculum(env)

		dwa_action = env.get_cmd_vel_dwa()
		# Select action randomly or according to policy
		if t < args.start_timesteps - 10e9:
			# Random around move_base values
			if t < 2e3:
				#action = env.random_action()
				action = env.randomize_action(dwa_action)
				#action = dwa_action
			else:
				action = env.randomize_action(dwa_action)
				#action = dwa_action
				#action = dwa_action
			#action = np.asarray([0,-0.3])
		else:
			action = (
				#policy.select_action(np.array(state))
				#policy.select_action(np.array(state), goal)
				policy.select_action(np.array(state), vel, goal)
				+ np.random.normal(0, max_action * args.expl_noise, size=action_dim)
			).clip(-max_action, max_action)

			action = (
				policy.select_action(np.array(state), vel, goal))

		# Perform action
		#next_state, reward, done = env.step_cdrl_short(action)
		#next_goal, next_state, reward, done = env.step_cdrl_short(action)
		joystick = env.get_cmd_vel()
		#action = np.zeros(2)
		action_joystick = np.zeros(2)
		action_joystick[0] = joystick.linear.x
		action_joystick[1] = joystick.angular.z
		#print(f"joystick: {action_joystick}")
		next_vel, next_goal, next_state, reward, done = env.step_cdrl_short_wp(action)
		#print(f"next_state: {next_state.shape}")

		if episode_timesteps > max_time_steps:
			reward = -1
			done = 1

		done_bool = float(done) if episode_timesteps < args.max_time_per_episode else 0 

		# Store data in replay buffer
		#replay_buffer.add(state, action, next_state, reward, done_bool)
		# replay_buffer.add(state, action, dwa_action, next_state, reward, done_bool)
		#normalized_action = (action + 1.5)/3
		#replay_buffer.add(goal, state, action, dwa_action, next_goal, next_state, reward, done_bool)
		replay_buffer.add(vel, goal, state, action, dwa_action, next_vel, next_goal, next_state, reward, done_bool)

		if episode_timesteps >= 4:
			if episode_timesteps == 4:
				print("Start recording!!!")
			#record(dir_path + "/DPO/dpo_simulation3/", vel, goal, state, action_joystick)
			pass


		vel = next_vel
		goal = next_goal
		state = next_state
		episode_reward += reward

		# Train agent after collecting sufficient data
		critic_loss = -1
		actor_loss = -1
		if t >= args.start_timesteps:
			# if t == args.start_timesteps:
			# 	for i in range(1000):
			# 		print("training")
			# 		critic_loss, actor_loss = policy.train(replay_buffer, args.batch_size)
			critic_loss, actor_loss = policy.train(replay_buffer, args.batch_size)

			
			#####
			updated_params = policy.critic.state_dict()
			#print(updated_params)
			#print(updated_params)
			# Check if any parameters have changed
			#parameters_changed = any((initial_params[key] != updated_params[key]).any() for key in initial_params)
			#print("\nParameters Changed:", parameters_changed)

			# Check if gradients are being updated
			gradients_updated = any(param.grad is not None for param in policy.critic.parameters())
			#print("Gradients Updated:", gradients_updated)
			#####

		if done: 
			# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
			print(f"t:{t}")
			print(f"episode_num: {episode_num}")
			print(f"episode_timesteps: {episode_timesteps}")
			print(f"episode_reward: {episode_reward}")
			print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
			# Reset environment

			if t > 8.5e3:
				all_rewards = np.append(all_rewards, episode_reward)
				critic_losses = np.append(critic_losses, critic_loss)
				actor_losses = np.append(actor_losses, actor_loss)
				save_results(all_rewards, critic_losses, actor_losses)
				#replay_buffer.save(dir_path)
				
				if USE_WANDB:
					wandb.log({"episode_reward": episode_reward}, step=episode_num)
			
			vel, goal, state, done = env.reset()
			print("start Reset")
			time.sleep(2.0) # Wait for reset to finish
			print("Finsih reset")
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1 


		# Evaluate episode
		if args.save_model:
			if (t + 1) % args.eval_freq == 0:
				np.save(f"{dir_path}/results/cdrl/{file_name}", evaluations)
				policy.save(f"{dir_path}/models_TD3_cdrl/{file_name}", "current")
				eval = eval_policy(policy, env)


				if (eval > min_eval) and (eval != 0): 
					policy.save(f"{dir_path}/models_TD3_cdrl/{file_name}", "best_" + str(int(t/1e3)) + str(int(eval)))
					min_eval = eval
				
				# Update env params
				#update_env_params(episode_num, t, min_eval, replay_buffer.ptr, replay_buffer.size)

def update_env_params(episode_num, t, min_eval, ptr, size):

	config, ind, bsi = ruamel.yaml.util.load_yaml_guess_indent(open(dir_path + "/env/env_params.yaml"))

	config["episode_num"] = episode_num
	config["min_eval"] = int(min_eval)
	config["time_step"] = t
	config["buffer_ptr"] = ptr
	config["buffer_size"] = size

	yaml = ruamel.yaml.YAML()
	yaml.indent(mapping=ind, sequence=ind, offset=bsi) 
	with open(dir_path + "/env/env_params.yaml", 'w') as fp:
		yaml.dump(config, fp)

def get_env_params():
	with open(dir_path + "/env/env_params.yaml", 'r') as f:
		data = list(yaml.load_all(f, Loader=SafeLoader))

	episode_num = data[0]["episode_num"]
	min_eval = data[0]["min_eval"]
	time_step = data[0]["time_step"]
	ptr = data[0]["buffer_ptr"]
	size = data[0]["buffer_size"]

	return episode_num, min_eval, time_step, ptr, size

def load_results():

	rewards = np.load(dir_path + '/results/cdrl/rewards.npy', allow_pickle=True)
	critic_losses = np.load(dir_path + '/results/cdrl/critic_losses.npy', allow_pickle=True)
	actor_losses = np.load(dir_path + '/results/cdrl/actor_losses.npy', allow_pickle=True)

	return rewards, critic_losses,actor_losses

def save_results(all_rewards, critic_losses, actor_losses):

	np.save(dir_path + '/results/cdrl/rewards.npy', all_rewards)
	np.save(dir_path + '/results/cdrl/critic_losses.npy', critic_losses)
	np.save(dir_path + '/results/cdrl/actor_losses.npy', actor_losses)

def record(folder, vel, goal, state, joystick):
	global record_counter, buffer_len, ptr, vel_buffer, goal_buffer, state_buffer, joystick_buffer

	vel_buffer[ptr] = vel
	goal_buffer[ptr] = goal
	state_buffer[ptr] = state
	joystick_buffer[ptr] = joystick

	if ptr == buffer_len - 1:
		# print(f"Saving recording")
		# print(f"vel_buffer: {vel_buffer}")
		# print(f"joystick_buffer: {joystick_buffer}")
		#print(f"goal_buffer: {goal_buffer}")
        # #Show images #######################################
		
		# fig2, (ax11, ax22) = plt.subplots(1, 2)
		# fig2.suptitle('Images')
		
		# im1 = Image.fromarray(np.fliplr(state_buffer[0])*256)
		# imgplot = ax11.imshow(im1)
		# ax11.set_title("Decoded input")
		
		# im3 = Image.fromarray(state_buffer[15]*256)
		# imgplot = ax22.imshow(im3)
		# ax22.set_title("Decoded prediction")
		
		# plt.show(block=False)
		# plt.pause(1)
		# plt.close()
		# plt.show()

		np.save(folder + "vel_" + str(record_counter), vel_buffer)
		np.save(folder + "goal_" + str(record_counter), goal_buffer)
		np.save(folder + "state_" + str(record_counter), state_buffer)
		np.save(folder + "action_" + str(record_counter), joystick_buffer)

		ptr = 0
		record_counter += 1
	else:
		ptr += 1





if __name__ == '__main__':
    rospy.init_node('init_train')
    main()
