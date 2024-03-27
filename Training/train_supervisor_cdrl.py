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
from deap import base, creator, tools, algorithms
import random
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dir_path = os.path.dirname(os.path.realpath(__file__))

env = None
policy = None

enforce_bounds = [[0.1, 1], [0.1, 1], [0.1, 1], [0.1, 1]]

num_pop = 16 # lambda
num_gen = 17

counter_pop = 0
counter_gen = 0
results_gen = np.zeros([num_gen,num_pop,2])
print(f"results_gen: {results_gen.shape}")


###########################################################################################################################
###########################################################################################################################
def main():

	global env, policy, counter_gen, counter_pop
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG or OurDDPG)
	parser.add_argument("--env", default="PTDRL")          # OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--start_timesteps", default=8e3, type=int)# Time steps initial random policy is used 25e3 ### 50e3, 2e3
	parser.add_argument("--eval_freq", default=2e3, type=int)       # How often (time steps) we evaluate - Before 5e3, now 3e4 ### 25e3, 2e3
	parser.add_argument("--max_timesteps", default=1e7, type=int)   # Max time steps to run environment. Between 15.000 to 150.000 episodes
	parser.add_argument("--max_time_per_episode", default=2000, type=int)  # Max episodes to run environment
	parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise ### Before 0.1
	parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic - Before 256
	parser.add_argument("--discount", default=0.99)                 # Discount factor - before 0.99
	parser.add_argument("--tau", default=0.005)                     # Target network update rate
	parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update ### Before 0.2
	parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise ### Before 0.5
	parser.add_argument("--policy_freq", default=4, type=int)       # Frequency of delayed policy updates ### Before 2
	parser.add_argument("--save_model", default = False, action="store_true")        # Save model and optimizer parameters
	parser.add_argument("--load_model", default=True)                 # Model load file name, "" doesn't load, "default" uses file_name
	args, unknown = parser.parse_known_args()
	

	file_name = f"{args.policy}_{args.env}_{args.seed}"
	print("---------------------------------------")
	print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")



	env = task_env.PtdrlTaskEnv()

	# Set seeds
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	
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
	

	if args.load_model:
		policy_file = file_name
		policy.load(f"{dir_path}/models_TD3_cdrl/{policy_file}", "current")
		print("Loading Network!!!")

	###################################################
	###################################################
		
	# Do optimization
	# Create a multi-objective fitness class with two objectives to minimize
	creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))

	# Create an individual class with a list attribute to represent the solution
	creator.create("Individual", list, fitness=creator.FitnessMulti)

	# Define the problem
	toolbox = base.Toolbox()
	toolbox.register("attr_float", random.uniform, 0.1, 1.0)
	toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=4)
	toolbox.register("population", tools.initRepeat, list, toolbox.individual)


	# Register the objective functions and genetic operators
	toolbox.register("evaluate", fitness_fn)
	toolbox.register("mate", tools.cxBlend, alpha=0.5)
	toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
	#toolbox.register("mutate", tools.mutPolynomialBounded, low=0.1, up=1.0, eta=20.0, indpb=0.2) 
	toolbox.register("select", tools.selNSGA2)

	population = toolbox.population(n=16) #32 #8
	hall_of_fame = tools.HallOfFame(1)

	# Run NSGA-II with serial evaluation
	algorithms.eaMuPlusLambda(
		population,
		toolbox,
		mu=16, # 32 #8
		lambda_=16,#64 #8
		cxpb=0.7,
		mutpb=0.2,
		ngen=16, #8
		stats=None,
		halloffame=hall_of_fame,
		verbose=True
		#parallelize_evaluation=False  # Set parallelize_evaluation to False for serial evaluation
	)

	# Print the best individual and its variables
	best_individual = hall_of_fame[0]
	print("Best individual:", best_individual)
	print("Best variables:", best_individual)

	#### Show results in 2D graph ####
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	# Scatter plot for each generation
	for i in range(num_gen):
		is_all_zero = np.all(results_gen[i, :, 0] == 0)
		if is_all_zero:
			break
		x_values = results_gen[i, :, 0]
		y_values = results_gen[i, :, 1]
		z_values = [i] * num_pop  # Assign the Z-value based on the generation index

		print(f"Gen {i} x - Mean: {np.mean(x_values)} Std: {np.std(x_values)}")
		print(f"Gen {i} y - Mean: {np.mean(y_values)} Std: {np.std(y_values)}")
		ax.scatter(x_values, y_values, z_values, label=f'Generation {i}')

	ax.set_xlabel('X-axis')
	ax.set_ylabel('Y-axis')
	ax.set_zlabel('Generation')
	ax.set_title('3D Scatter Plot of Points across Generations')
	plt.legend()
	plt.show()
	#################################################
	#################################################

		

###########################################################################################################################
###########################################################################################################################

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def fitness_fn(x):
	global counter_pop, counter_gen
	print(f"x:{x}")
	x = np.clip(x, 0.1, 1.0)
	print(f"x clipped:{x}")
	#print("---------------------------------------")
	#print("Starting evaluation")
	#print("---------------------------------------")
	max_episodes = 100
	avg_supervisor = 0.
	avg_supervisor_general = 0.

	eval_episodes=1

	for _ in range(eval_episodes):
		vel, goal, state, done = env.reset()
		counter = 0
		while not done:
			counter += 1
			action = policy.select_action(state, vel, goal)
			action = env.randomize_action(action) # Randomized action
			supervisor, supervisor_general, vel, goal, state, reward, done = env.step_cdrl_train_supervisor_wp(action, x)
			avg_supervisor += supervisor
			avg_supervisor_general += supervisor_general

			if counter > max_episodes:
				done = True


	avg_supervisor /= eval_episodes
	avg_supervisor_general /= eval_episodes

	######################
	### Store results ####
	print(f"counter_gen: {counter_gen}")
	print(f"counter_pop: {counter_pop}")
	results_gen[counter_gen, counter_pop, 0] = avg_supervisor
	results_gen[counter_gen, counter_pop, 1] = avg_supervisor_general
	# Update counters
	counter_pop += 1
	if counter_pop == num_pop:
		counter_gen += 1
		counter_pop = 0
	######################
	######################


	print("---------------------------------------")
	print(f"Evaluation: {avg_supervisor:.3f}, {avg_supervisor_general:.3f}")
	print("---------------------------------------")
	return (avg_supervisor, avg_supervisor_general)


if __name__ == '__main__':
    rospy.init_node('init_train')
    main()
