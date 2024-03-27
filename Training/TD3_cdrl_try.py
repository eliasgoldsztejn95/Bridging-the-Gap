import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.simple_vit import SimpleViT
from models.simple_vit_reward import SimpleViTReward
from models.cnn_goal import CNNWithGoal, CNNWithGoalReward, CNNWithGoalTiny, CNNWithGoalRewardTiny, CNNActor, CNNCritic, ActorCNNFine, CriticCNNFine, FCActor, FCCritic, CNNActorV2, CNNCriticV2, CNNActorV3, CNNCriticV3, CNNActorV4, CNNCriticV4, CNNActorV6, CNNCriticV6

from PIL import Image
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		# self._actor = SimpleViT(
		# 	image_size=(state_dim, state_dim),
		# 	patch_size=(6, 6),
		# 	dim=128, #128
		# 	depth=4, #4
		# 	heads=4, #2
		# 	mlp_dim=256, #128
		# 	output_dim=action_dim
		# ).to(device)

		#self._actor = CNNWithGoal().to(device)

		self._actor = CNNActorV3().to(device) # Winnning

		#self._actor = ActorCNNFine().to(device)

		#self._actor = FCActor().to(device)
		
		self.max_action = max_action
		

	#def forward(self, state):
	def forward(self, state, vel, goal):
		#### state_tensor = torch.FloatTensor(state).to(device)

		#return self.max_action * torch.tanh(self._actor(state))
		return self.max_action * torch.tanh(self._actor(state, vel, goal))

class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		# Q1 architecture
		# self._q1 = SimpleViTReward(
		# 	image_size=(state_dim, state_dim),
		# 	patch_size=(6, 6),
		# 	dim=128, #128
		# 	depth=4, #4
		# 	heads=4, #3
		# 	mlp_dim=256, #256
		# 	output_dim=1
		# ).to(device)

		#self._q1 = CNNWithGoalReward().to(device)

		self._q1 = CNNCriticV3().to(device)  ## Winning

		#self._q1 = CriticCNNFine().to(device)

		#self._q1 = FCCritic().to(device)

		# Q2 architecture
		# self._q2 = SimpleViTReward(
		# 	image_size=(state_dim, state_dim),
		# 	patch_size=(6, 6),
		# 	dim=128, 
		# 	depth=4, 
		# 	heads=4,
		# 	mlp_dim=256,
		# 	output_dim=1
		# ).to(device)

		#self._q2 = CNNWithGoalReward().to(device)

		self._q2 = CNNCriticV3().to(device)  ## Winning

		#self._q2 = CriticCNNFine().to(device)

		#self._q2 = FCCritic().to(device) 


	#def forward(self, state, action):
	def forward(self, state, vel, goal, action):
		sa = state#self.add_action_to_costmap(state, action)
		v = vel
		g = goal

		# q1 = self._q1(sa, action)

		# q2 = self._q2(sa, action)

		q1 = self._q1(sa, v, g, action)

		q2 = self._q2(sa, v, g, action)

		return q1, q2


	def Q1(self, state, vel, goal, action):
		sa = state#self.add_action_to_costmap(state, action)
		v = vel
		g = goal

		#q1 = self._q1(sa, action)
		q1 = self._q1(sa, v, g,  action)

		return q1



class TD3(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		max_action,
		discount=0.9, #0.9
		tau=0.005,
		policy_noise=0.2, # 0.2
		noise_clip=0.5, # 0.5
		policy_freq=16 # 4
	):

		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4) # Before self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)# Before self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq

		self.total_it = 0


	def select_action(self, state, vel, goal):
		#state = torch.cuda.FloatTensor(state.reshape(1, -1)).to(device)

		# goal = torch.cuda.FloatTensor(goal).to(device)
		# goal = goal.unsqueeze(0)
		# img = torch.cuda.FloatTensor(state).to(device)
		# img = img.unsqueeze(0)
		# img = img.unsqueeze(0)
		vel = torch.cuda.FloatTensor(vel).to(device)
		vel = vel.unsqueeze(0)
		goal = torch.cuda.FloatTensor(goal).to(device)
		goal = goal.unsqueeze(0)
		img = torch.cuda.FloatTensor(state).to(device)
		img = img.unsqueeze(0)
		img = img.unsqueeze(0)
		return self.actor(img, vel, goal).cpu().data.numpy().flatten()


	def train(self, replay_buffer, batch_size=256): #256 !!!!!!!!!!!!!
		self.total_it += 1

		# Sample replay buffer 
		#state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
		#state, action, dwa_action, next_state, reward, not_done = replay_buffer.sample(batch_size)
		vel, goal, state, action, dwa_action, next_vel, next_goal, next_state, reward, not_done = replay_buffer.sample(batch_size)

		with torch.no_grad():
			# Select action according to policy and add clipped noise
			noise = (
				torch.randn_like(action) * self.policy_noise
			).clamp(-self.noise_clip, self.noise_clip)
			
			next_action = (
				self.actor_target(next_state, next_vel, next_goal) + noise
			).clamp(-self.max_action, self.max_action)

			# Compute the target Q value
			target_Q1, target_Q2 = self.critic_target(next_state, next_vel, next_goal, next_action)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = reward + not_done * self.discount * target_Q

		# Get current Q estimates
		current_Q1, current_Q2 = self.critic(state, vel, goal, action)

		if self.total_it % 100 == 0:
			print(f"Current Q1 {current_Q1[0]}")
			print("!!!!!!!!!!!!!!!!!!!")
			print(f"Current Q2 {current_Q2[0]}")
			print("!!!!!!!!!!!!!!!!!!!")
			print(f"Current Target {target_Q[0]}")
			print("!!!!!!!!!!!!!!!!!!!")

		# Compute critic loss
		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
		#print(f"critic_loss: {critic_loss}")

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		#for param in self.critic.parameters():
			#print('Gradient:', param.grad)
		self.critic_optimizer.step()

		actor_loss = None
		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:

			
			#print(f"state: {state}")
			#actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

			# Clip the MSE loss
			threshold = 0.2  # Set your desired threshold

			non_zero_mask = torch.all(torch.ne(dwa_action, torch.tensor([0.0, 0.0]).to(device)), dim=1) # Take only places where dwa_action is not [0,0]

			non_zero_linear_mask = (dwa_action[:, 0] != 0.0)

			mse_loss_mask = F.mse_loss(self.actor(state, vel, goal)[non_zero_linear_mask], dwa_action[non_zero_linear_mask])
			
			#clipped_mse_loss = torch.clamp(mse_loss, min=threshold)

			##print(f"MSE actor dwa: {mse_loss} --- {F.mse_loss(self.actor(state), dwa_action)}")		
			##clipped_mse_loss = torch.clamp(F.mse_loss(self.actor(state), dwa_action), min=threshold) # Many dwa_actions are 0. We do not want to copy this behavior

			# Compute actor loss
			#print(dwa_action)

			#actor_loss = -self.critic.Q1(state, vel, goal, self.actor(state, vel, goal)).mean() + 0*mse_loss_mask + 0*F.mse_loss(self.actor(state, vel, goal), dwa_action) # CDRL loss

			#################
			### DPO #########

			pi = self.actor(state, vel, goal)
			pi_tag = self.actor_target(state, vel, goal)
			#lam = 1/self.critic.Q1(state, vel, goal, pi).abs().mean().detach()

			actor_loss = -self.critic.Q1(state, vel, goal, pi).mean() - F.logsigmoid(-F.mse_loss(pi[non_zero_linear_mask], dwa_action[non_zero_linear_mask]) + 
																			F.mse_loss(pi_tag[non_zero_linear_mask], dwa_action[non_zero_linear_mask]) + 
																			F.mse_loss(pi_tag[non_zero_linear_mask], pi[non_zero_linear_mask])) # DPO loss

			#print(f"MSE DPO loss: {(-F.mse_loss(pi[non_zero_linear_mask], dwa_action[non_zero_linear_mask]) + F.mse_loss(pi_tag[non_zero_linear_mask], dwa_action[non_zero_linear_mask]) + F.mse_loss(pi_tag[non_zero_linear_mask], pi[non_zero_linear_mask]))}")
			#print(f"DPO loss: {-F.logsigmoid(-F.mse_loss(pi[non_zero_linear_mask], dwa_action[non_zero_linear_mask]) + F.mse_loss(pi_tag[non_zero_linear_mask], dwa_action[non_zero_linear_mask]) + F.mse_loss(pi_tag[non_zero_linear_mask], pi[non_zero_linear_mask]))}")
			#print(f"(pi,dwa): {-F.mse_loss(pi[non_zero_linear_mask], dwa_action[non_zero_linear_mask])}, (pi_tag,dwa): {F.mse_loss(pi_tag[non_zero_linear_mask], dwa_action[non_zero_linear_mask])}, (pi_tag, pi): {F.mse_loss(pi_tag[non_zero_linear_mask], pi[non_zero_linear_mask])}")
			#################
			#################

			#print(f"Q_loss: {lam*self.critic.Q1(state, vel, goal, pi).mean()}")
			#print(f"actor dwa: {F.mse_loss(torch.zeros((256, 2)).to(device), dwa_action)}")

			if self.total_it % 100 == 0:
				#print(self.total_it)
				#print(f"Q1: {self.critic.Q1(state, vel, goal, self.actor(state, vel, goal)).mean()}")
				print(f"actor dwa: {F.mse_loss(self.actor(state, vel, goal), dwa_action)}")
				#print(f"mse_loss_mask: {mse_loss_mask}")
				#print(f"actor: {self.actor(state, vel, goal)[0:10]}")
				#print(f"dwa: {dwa_action[0:10]}")
				#print(f"dwa: {dwa_action[0:10][non_zero_linear_mask[0:10]]}")
			#print(dwa_action)

			#print(f"actor_loss: {actor_loss}")
			
			# Optimize the actor 
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			#for param in self.actor.parameters():
				#print('Gradient:', param.grad)
			self.actor_optimizer.step()

			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		if actor_loss is not None:
			return critic_loss.cpu().data.numpy(), actor_loss.cpu().data.numpy()
		return critic_loss.cpu().data.numpy(), -1


	def save(self, filename, current_best):
		torch.save(self.critic.state_dict(), filename + "_critic_" + current_best)
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer_" + current_best)
		
		torch.save(self.actor.state_dict(), filename + "_actor_" + current_best)
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer_" + current_best)


	def load(self, filename, current_best):
		self.critic.load_state_dict(torch.load(filename + "_critic_" + current_best))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer_" + current_best))
		self.critic_target = copy.deepcopy(self.critic)

		self.actor.load_state_dict(torch.load(filename + "_actor_" + current_best))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer_" + current_best))
		self.actor_target = copy.deepcopy(self.actor)

	def load_dpo(self, filename, filename_actor):
		self.critic.load_state_dict(torch.load(filename + "_critic_current"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer_current"))
		self.critic_target = copy.deepcopy(self.critic)

		self.actor.load_state_dict(torch.load(filename_actor))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer_current"))
		self.actor_target = copy.deepcopy(self.actor)
		