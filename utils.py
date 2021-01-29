import numpy as np
import torch
import os
from torch.utils.tensorboard import SummaryWriter


class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6), device=torch.device('cuda')):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.device = device


	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)
	
class ReplayBufferTorch(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6), device=torch.device('cuda'), discount=0.99):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0
		self.discount = discount
		self.state = torch.zeros((max_size, state_dim), device=device)
		self.action = torch.zeros((max_size, action_dim), device=device)
		self.next_state = torch.zeros((max_size, state_dim), device=device)
		self.reward = torch.zeros((max_size, 1), device=device)
		self.not_done = torch.zeros((max_size, 1), device=device)
		self.max_step = np.ones((max_size,1))
		self.return_list = torch.zeros((max_size,1),device=device)
		self.device = device
		self.acc = 0
		self.no_done = False
		self.start = True
		self.reward_min = 0
		self.reward_min_index = 0
		
	def add(self, state, action, next_state, reward, done, step, min_step, max_step):
		self.state[self.ptr] = torch.tensor(state, device=self.device)
		self.action[self.ptr] = torch.tensor(action, device=self.device)
		self.next_state[self.ptr] = torch.tensor(next_state, device=self.device)
		self.reward[self.ptr] = torch.tensor(reward, device=self.device)
		self.max_step[self.ptr] = step
		if self.no_done:	
			if self.start and self.ptr == 999:
				self.reward_min_index = torch.argmin(self.reward[:1000])
				self.reward_min = self.reward[self.reward_min_index]
				self.return_list[self.ptr] = self.reward_min/(1 - self.discount)
				self.start = False
			else:
				if self.ptr > 999 or (not self.start):
					if self.reward[self.ptr] < self.reward_min:
						self.reward_min = self.reward[self.ptr]
						self.reward_min_index = self.ptr
					if ((self.ptr - 1000) % self.max_size) == self.reward_min_index:
						if self.ptr-999 >= 0 and self.ptr+1 <= self.max_size:
							self.reward_min_index = torch.argmin(self.reward[self.ptr-999:self.ptr+1])
							self.reward_min = self.reward[self.reward_min_index]
						elif self.ptr - 999 < 0:
							self.reward_min_index1 = torch.argmin(self.reward[0:self.ptr+1])
							self.reward_min_index2 = torch.argmin(self.reward[self.ptr-999-1:])
							self.reward_min_1 = self.reward[self.reward_min_index1]
							self.reward_min_2 = self.reward[self.reward_min_index2]
							if self.reward_min_1 > self.reward_min_2:
								self.reward_min = self.reward_min_2
								self.reward_min_index = self.reward_min_index2
							else:
								self.reward_min = self.reward_min_1
								self.reward_min_index = self.reward_min_index1
					self.return_list[self.ptr] = self.reward_min / (1-self.discount)
		else: 
			#if done > 0.9:
			#	ptr_cur = self.ptr
			#	ptr_prev = (ptr_cur - 1) % self.max_size
			#	while self.not_done[ptr_prev] > 0.9:
		#			self.max_step[ptr_prev] = step
	#				ptr_cur = ptr_prev
#					ptr_prev = (ptr_cur - 1) % self.max_size
			if self.start and self.ptr == 999:
				self.reward_min_index = torch.argmin(self.reward[:1000])
				self.reward_min = self.reward[self.reward_min_index]
				if self.reward_min > 0:
					self.return_list[self.ptr] = self.reward_min/(1 - self.discount) * (1 - self.discount ** min_step)
				else:
					self.return_list[self.ptr] = self.reward_min/(1 - self.discount) * (1 - self.discount ** max_step)
				self.start = False
			else:
				if self.ptr > 999 or (not self.start):
					if self.reward[self.ptr] < self.reward_min:
						self.reward_min = self.reward[self.ptr]
						self.reward_min_index = self.ptr
					if ((self.ptr - 1000) % self.max_size) == self.reward_min_index:
						if self.ptr - 999 >= 0 and self.ptr+1 <= self.max_size:
							self.reward_min_index = torch.argmin(self.reward[self.ptr-999:self.ptr+1])
							self.reward_min = self.reward[self.reward_min_index]
						elif self.ptr - 999 < 0:
							self.reward_min_index1 = torch.argmin(self.reward[0:self.ptr+1])
							self.reward_min_index2 = torch.argmin(self.reward[self.ptr-999-1:])
							self.reward_min_1 = self.reward[self.reward_min_index1]
							self.reward_min_2 = self.reward[self.reward_min_index2]
							if self.reward_min_1 > self.reward_min_2:
								self.reward_min = self.reward_min_2
								self.reward_min_index = self.reward_min_index2
							else:
								self.reward_min = self.reward_min_1
								self.reward_min_index = self.reward_min_index1
					if self.reward_min > 0:
						self.return_list[self.ptr] = self.reward_min/(1 - self.discount) * (1 - self.discount ** min_step)
					else:
						self.return_list[self.ptr] = self.reward_min/(1 - self.discount) * (1 - self.discount ** max_step)
			
		self.not_done[self.ptr] = torch.tensor(1. - done, device=self.device)
		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)

	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)
		return (
			self.state[ind],
			self.action[ind],
			self.next_state[ind],
			self.reward[ind],
			self.not_done[ind],
			self.return_list[ind],
		)
	# def save(self, log_dir):
	# 	import pickle
	# 	with open('{}/buffer_{}.pkl'.format(log_dir, self.ptr), 'wb+') as f:
	# 		pickle.dump({
	# 			'state': self.state,
	# 			'action': self.action,
	# 			'next_state': self.next_state,
	# 			'reward': self.reward,
	# 			'not_done': self.not_done
	# 		}, f)

class WriterLoggerWrapper(object):
	def __init__(self, log_dir, comment, max_timesteps):
		self.tf_writer = SummaryWriter(log_dir=log_dir, comment=comment)
		
		logger_result_path = '{}/{}'.format(log_dir, 'log_txt')
		if not os.path.exists(logger_result_path):
			os.makedirs(logger_result_path)
		print(logger_result_path)
		self.logger = Logger(logger_result_path, max_timesteps)

	def add_scalar(self, scalar_name, scalar_val, it):
		self.tf_writer.add_scalar(scalar_name, scalar_val, it)
		self.logger.add_scalar(scalar_name, scalar_val, it)

class Logger(object):
	def __init__(self, log_dir, max_timesteps):
		self.log_dir = log_dir
		self.max_timesteps = max_timesteps
		self.all_data = {}

	def add_scalar(self, scalar_name, scalar_val, it):
		if not (scalar_name in self.all_data.keys()):
			# add new entry
			self.all_data[scalar_name] = np.zeros([int(self.max_timesteps + 1)])
		self.all_data[scalar_name][int(it)] = scalar_val
	
	def save_to_txt(self, log_dir=None):
		if log_dir is None:
			log_dir = self.log_dir
		
		for tag in self.all_data.keys():
			np.savetxt('{}/{}data.txt'.format(log_dir, tag.replace('/', '_')), self.all_data[tag], delimiter='\n', fmt='%.5f')
