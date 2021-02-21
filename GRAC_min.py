import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ES import Searcher
from torch.distributions import Normal
import math
import matplotlib.pyplot as plt
import datetime
import os
# Implementation of Self-Guided and Self-Regularized Actor-Critic Algorithm.

epsilon = 1e-6

class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()
		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3_sigma = nn.Linear(256, action_dim)
		self.l3_mean = nn.Linear(256, action_dim)
		
		self.max_action = max_action
		self.action_dim = action_dim
		self.state_dim = state_dim

	def forward(self, state, *args):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		mean = self.max_action * torch.tanh(self.l3_mean(a))
		sigma = F.softplus(self.l3_sigma(a)) + 0.001
		normal = Normal(mean, sigma)
		action = normal.rsample().clamp(-self.max_action, self.max_action)
		return action

	def forward_all(self, state, *args):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		mean = self.max_action * torch.tanh(self.l3_mean(a))
		sigma = F.softplus(self.l3_sigma(a)) + 0.001
		normal = Normal(mean, sigma)
		action = normal.rsample()
		log_prob = normal.log_prob(action).sum(1,keepdim=True)
		action = action.clamp(-self.max_action, self.max_action)
		return action, log_prob, mean, sigma

class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim, 256)
		self.l5 = nn.Linear(256, 256)
		self.l6 = nn.Linear(256, 1)


	def forward(self, state, action):

		sa = torch.cat([state, action], len(action.shape)-1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2


	def Q1(self, state, action):
		sa = torch.cat([state, action], len(action.shape)-1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1

	def Q2(self, state, action):
		sa = torch.cat([state, action], len(action.shape)-1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q2

	def Q12(self, state, action):
		Q1 = self.Q1(state, action)
		Q2 = self.Q2(state, action)
		Q = torch.min(Q1, Q2)
		return Q

class GRAC():
	def __init__(
		self,
		state_dim,
		action_dim,
		max_action,
		batch_size=256,
		discount=0.99,
		max_timesteps=3e6,
                actor_lr = 3e-4,
                critic_lr = 3e-4,
                loss_decay = 0.95,
                log_freq = 200,
		actor_lr_ratio =1.0,
		device=torch.device('cuda'),
	):
		self.action_dim = action_dim
		self.state_dim = state_dim
		self.max_action = max_action
		self.discount = discount
		self.total_it = 0

		self.device = device
		self.actor_lr = actor_lr # here is actor lr is not the real actor learning rate
		self.critic_lr = critic_lr
		self.loss_decay = loss_decay
		self.actor_lr_ratio = actor_lr_ratio

		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)

		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)

		cem_sigma = 1e-2
		cem_clip = 0.5 * max_action
		self.cem_clip_init = cem_clip
		self.cem_clip = cem_clip
		self.searcher = Searcher(action_dim, max_action, device=device, sigma_init=cem_sigma, clip=cem_clip, batch_size=batch_size)
		self.action_dim = float(action_dim)
		self.log_freq = log_freq
		self.max_timesteps = max_timesteps
		self.cem_loss_coef = 1.0/float(self.action_dim)
		self.selection_action_coef = 1.0


	def select_action(self, state, writer=None, test=False):
		state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
		if test is False:
			with torch.no_grad():
				action = self.actor(state)
				better_action = action
			return better_action.cpu().data.numpy().flatten()

		else:
			_, _, action, _ = self.actor.forward_all(state)
			return action.cpu().data.numpy().flatten()

	def lr_scheduler(self, optimizer,lr):
		for param_group in optimizer.param_groups:
			param_group['lr'] = lr
		return optimizer

	def update_critic(self, critic_loss):
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))

	def train(self, replay_buffer, batch_size=100, writer=None, reward_range=20.0, reward_max=0, episode_step_max=100, reward_min=0, episode_step_min=1):
		self.total_it += 1
		log_it = (self.total_it % self.log_freq == 0)
		ratio_it = max(1.0 - self.total_it/float(self.max_timesteps), 0.1)
		if log_it:
			writer.add_scalar('train_critic/ratio_it', ratio_it, self.total_it)
		self.cem_clip = self.cem_clip_init * ratio_it
		# Sample replay buffer 
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
		with torch.no_grad():
			# Select action according to policy and add clipped noise
			next_action = (
				self.actor(next_state)
			).clamp(-self.max_action, self.max_action)
			#better_next_action,_ = self.searcher.search(next_state, next_action, self.critic.Q2, clip=self.cem_clip)
			target_Q1, target_Q2 = self.critic(next_state, next_action)
			target_Q = torch.min(target_Q1, target_Q2)

			#better_target_Q1, better_target_Q2 = self.critic(next_state, better_next_action)
			#better_target_Q = torch.min(better_target_Q1, better_target_Q2)

			#action_index = (target_Q > better_target_Q).squeeze()
			#better_next_action[action_index] = next_action[action_index]
			#better_target_Q1, better_target_Q2 = self.critic(next_state, better_next_action)

			better_target_Q = target_Q#torch.min()#torch.max(better_target_Q, target_Q)

			#target_Q1 = better_target_Q1
			#target_Q2 = better_target_Q2

			Q_max = reward_max / (1 - self.discount) * (1 - self.discount ** int(episode_step_max))
			target_Q1[target_Q1 > Q_max] = Q_max
			target_Q2[target_Q2 > Q_max] = Q_max
			better_target_Q[better_target_Q > Q_max] = Q_max
			if reward_min >= 0:
				Q_min = reward_min / (1 - self.discount) * (1 - self.discount ** int(episode_step_min))
			else:
				Q_min = reward_min / (1 - self.discount) * (1 - self.discount ** int(episode_step_max))
			target_Q1[target_Q1 < Q_min] = Q_min
			target_Q2[target_Q2 < Q_min] = Q_min
			better_target_Q[better_target_Q < Q_min] = Q_min

			target_Q_final = reward + not_done * self.discount * better_target_Q
			target_Q_final[target_Q_final > Q_max] = Q_max
			target_Q_final[target_Q_final < Q_min] = Q_min
			#next_action = better_next_action

		# Get current Q estimates
		current_Q1, current_Q2 = self.critic(state, action)

		# Compute critic loss
		critic_loss = F.mse_loss(current_Q1, target_Q_final) + F.mse_loss(current_Q2, target_Q_final)
		self.update_critic(critic_loss)

		current_Q1_, current_Q2_ = self.critic(state, action)
		target_Q1_, target_Q2_ = self.critic(next_state, next_action)
		critic_loss2_1 = F.mse_loss(current_Q1_, target_Q_final) + F.mse_loss(current_Q2_, target_Q_final) 
		critic_loss2_2 = F.mse_loss(target_Q1_, target_Q1) + F.mse_loss(target_Q2_, target_Q2)	
		weight1 = critic_loss2_1.item()
		weight2 = critic_loss2_2.item()	
		weight_loss = (math.sqrt(weight1) + 1.)/( math.sqrt(weight2) + 1.)
		critic_loss3 = critic_loss2_1 + critic_loss2_2 * weight_loss
		self.update_critic(critic_loss3)
		init_critic_loss3 = critic_loss3.clone()
		ratio = 0.0
		max_step = 0

		idi = 0
		while True:
			idi = idi + 1
			current_Q1_, current_Q2_ = self.critic(state, action)
			target_Q1_, target_Q2_ = self.critic(next_state, next_action)
			critic_loss3_1 = F.mse_loss(current_Q1_, target_Q_final) + F.mse_loss(current_Q2_, target_Q_final)
			critic_loss3_2 = F.mse_loss(target_Q1_, target_Q1) + F.mse_loss(target_Q2_, target_Q2)
			critic_loss3 = critic_loss3_1 + critic_loss3_2 * weight_loss
			self.update_critic(critic_loss3)
			if critic_loss3_1 < critic_loss * self.loss_decay and critic_loss3_1 < critic_loss2_1 * self.loss_decay and torch.sqrt(critic_loss3_2) < torch.max(torch.mean(torch.abs(better_target_Q)) * 0.01, torch.mean(torch.abs(reward))) and torch.sqrt(critic_loss3_2) < reward_max * 1.0:
                                        break
			if idi > 50:
				break
		#critic_loss = F.mse_loss(current_Q1, target_Q_final) + F.mse_loss(current_Q2, target_Q_final)
		weights_actor_lr = critic_loss2_1.item()
		if log_it:
			writer.add_scalar('train_critic/weight_loss', weight_loss, self.total_it)
			writer.add_scalar('train_critic/third_loss_num', idi, self.total_it)
			writer.add_scalar('train_critic/Q_max', Q_max, self.total_it)
			writer.add_scalar('train_critic/episode_step_max',episode_step_max, self.total_it)
			writer.add_scalar('train_critic/Q_min', Q_min, self.total_it)
			writer.add_scalar('train_critic/cem_clip', self.cem_clip, self.total_it)
			#writer.add_scalar('train_critic/Q_min_mean', torch.mean(Q_min), self.total_it)
			#writer.add_scalar('train_critic/Q_min_min', torch.min(Q_min), self.total_it)
			writer.add_scalar('train_critic/episode_step_min',episode_step_min, self.total_it)
		if log_it:
			writer.add_scalar('train_loss/loss2_1',critic_loss2_1,self.total_it)
			writer.add_scalar('train_loss/loss2_2',critic_loss2_2,self.total_it)
			writer.add_scalar('train_loss/loss3_1_r',critic_loss3_1/critic_loss2_1,self.total_it)
			writer.add_scalar('train_loss/loss3_2_r',critic_loss3_2/critic_loss2_2,self.total_it)
			writer.add_scalar('train_loss/loss3_1_r_loss',critic_loss3_1/critic_loss,self.total_it)
			writer.add_scalar('train_loss/sqrt_critic_loss3_2',torch.sqrt(critic_loss3_2),self.total_it)
			writer.add_scalar('train_loss/targetQ_condition',torch.mean(torch.abs(better_target_Q)) * 0.01,self.total_it)
			writer.add_scalar('train_loss/reward_condition',torch.mean(torch.abs(reward)),self.total_it)
			writer.add_scalar('train_loss/max_reward',reward_max,self.total_it)	
			#writer.add_scalar('train_loss/min_reward',reward_min,self.total_it)
		if self.total_it % 1 == 0:
			lr_tmp = self.actor_lr / (float(weights_actor_lr)+1.0) * self.actor_lr_ratio
			self.actor_optimizer = self.lr_scheduler(self.actor_optimizer, lr_tmp)

			# Compute actor loss
			actor_action, log_prob, action_mean, action_sigma = self.actor.forward_all(state)
			q_actor_action = self.critic.Q12(state, actor_action)
			m = Normal(action_mean, action_sigma)

			better_action,_ = self.searcher.search(state, actor_action, self.critic.Q12, batch_size=batch_size, clip=self.cem_clip)#####
			q_better_action = self.critic.Q12(state, better_action)
			log_prob_better_action = m.log_prob(better_action).sum(1,keepdim=True)

			adv = (q_better_action - q_actor_action).detach()
			adv = torch.max(adv,torch.zeros_like(adv))
			cem_loss = log_prob_better_action * torch.min(reward_range * torch.ones_like(adv) * ratio_it, adv)
			actor_loss = -(cem_loss * self.cem_loss_coef + q_actor_action).mean()

			# Optimize the actor 
			#Q_before_update = self.critic.Q1(state, actor_action)

			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()
			self.actor_optimizer = self.lr_scheduler(self.actor_optimizer, self.actor_lr)

		if log_it:
			with torch.no_grad():
				writer.add_scalar('train_critic/third_loss_num', idi, self.total_it)
				writer.add_scalar('train_critic/critic_loss', critic_loss, self.total_it)
				writer.add_scalar('train_critic/critic_loss3', critic_loss3, self.total_it)
	
				target_Q1_Q2_diff = target_Q1 - target_Q2
				writer.add_scalar('q_diff/target_Q1_Q2_diff_max', target_Q1_Q2_diff.max(), self.total_it)
				writer.add_scalar('q_diff/target_Q1_Q2_diff_min', target_Q1_Q2_diff.min(), self.total_it)
				writer.add_scalar('q_diff/target_Q1_Q2_diff_mean', target_Q1_Q2_diff.mean(), self.total_it)
				writer.add_scalar('q_diff/target_Q1_Q2_diff_abs_mean', target_Q1_Q2_diff.abs().mean(), self.total_it)
	
				current_Q1_Q2_diff = current_Q1 - current_Q2
				writer.add_scalar('q_diff/current_Q1_Q2_diff_max', current_Q1_Q2_diff.max(), self.total_it)
				writer.add_scalar('q_diff/current_Q1_Q2_diff_min', current_Q1_Q2_diff.min(), self.total_it)
				writer.add_scalar('q_diff/current_Q1_Q2_diff_mean', current_Q1_Q2_diff.mean(), self.total_it)
				writer.add_scalar('q_diff/current_Q1_Q2_diff_abs_mean', current_Q1_Q2_diff.abs().mean(), self.total_it)
	
				#target_Q1
				writer.add_scalar('train_critic/target_Q1/mean', torch.mean(target_Q1), self.total_it)
				writer.add_scalar('train_critic/target_Q1/max', target_Q1.max(), self.total_it)
				writer.add_scalar('train_critic/target_Q1/min', target_Q1.min(), self.total_it)
				writer.add_scalar('train_critic/target_Q1/std', torch.std(target_Q1), self.total_it)
				#target_Q2
				writer.add_scalar('train_critic/target_Q2/mean', torch.mean(target_Q2), self.total_it)
	
				#current_Q1
				writer.add_scalar('train_critic/current_Q1/mean', current_Q1.mean(), self.total_it)
				writer.add_scalar('train_critic/current_Q1/std', torch.std(current_Q1), self.total_it)
				writer.add_scalar('train_critic/current_Q1/max', current_Q1.max(), self.total_it)
				writer.add_scalar('train_critic/current_Q1/min', current_Q1.min(), self.total_it)
	
				# current_Q2
				writer.add_scalar('train_critic/current_Q2/mean', current_Q2.mean(), self.total_it)

	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
