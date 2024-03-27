import numpy as np
import torch


class ReplayBufferGoal(object):
	def __init__(self, state_dim, action_dim, max_size=int(2e5)): # - Before 1e6
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.vel = np.zeros((max_size, 2))
		self.goal = np.zeros((max_size, 2))
		self.state = np.zeros((max_size, state_dim, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.dwa_action = np.zeros((max_size, action_dim))
		self.next_vel = np.zeros((max_size, 2))
		self.next_goal = np.zeros((max_size, 2))
		self.next_state = np.zeros((max_size, state_dim, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def add(self, vel, goal, state, action, dwa_action, next_vel, next_goal, next_state, reward, done):
		self.vel[self.ptr] = vel
		self.goal[self.ptr] = goal
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.dwa_action[self.ptr] = dwa_action
		self.next_vel[self.ptr] = next_vel
		self.next_goal[self.ptr] = next_goal
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.vel[ind]).to(self.device),
			torch.FloatTensor(self.goal[ind]).to(self.device),
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.dwa_action[ind]).to(self.device),
			torch.FloatTensor(self.next_vel[ind]).to(self.device),
			torch.FloatTensor(self.next_goal[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)
	
	def save(self, dir_path):
		np.save(dir_path + "/models_TD3_cdrl/buffer_vel.npy", self.vel)
		np.save(dir_path + "/models_TD3_cdrl/buffer_goal.npy", self.goal)
		np.save(dir_path + "/models_TD3_cdrl/buffer_state.npy", self.state)
		np.save(dir_path + "/models_TD3_cdrl/buffer_action.npy", self.action)
		np.save(dir_path + "/models_TD3_cdrl/buffer_dwa_action.npy", self.dwa_action)
		np.save(dir_path + "/models_TD3_cdrl/buffer_next_vel.npy", self.next_vel)
		np.save(dir_path + "/models_TD3_cdrl/buffer_next_goal.npy", self.next_goal)
		np.save(dir_path + "/models_TD3_cdrl/buffer_next_state.npy", self.next_state)
		np.save(dir_path + "/models_TD3_cdrl/buffer_reward.npy", self.reward)
		np.save(dir_path + "/models_TD3_cdrl/buffer_not_done.npy", self.not_done)

	def load(self, dir_path):
		self.goal = np.load(dir_path + "/models_TD3_cdrl/buffer_vel.npy")
		self.goal = np.load(dir_path + "/models_TD3_cdrl/buffer_goal.npy")
		self.state = np.load(dir_path + "/models_TD3_cdrl/buffer_state.npy")
		self.action = np.load(dir_path + "/models_TD3_cdrl/buffer_action.npy")
		self.dwa_action = np.load(dir_path + "/models_TD3_cdrl/buffer_dwa_action.npy")
		self.goal = np.load(dir_path + "/models_TD3_cdrl/buffer_next_vel.npy")
		self.next_goal = np.load(dir_path + "/models_TD3_cdrl/buffer_next_goal.npy")
		self.next_state = np.load(dir_path + "/models_TD3_cdrl/buffer_next_state.npy")
		self.reward = np.load(dir_path + "/models_TD3_cdrl/buffer_reward.npy")
		self.not_done = np.load(dir_path + "/models_TD3_cdrl/buffer_not_done.npy")

