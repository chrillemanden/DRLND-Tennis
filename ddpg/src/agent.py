import random
import numpy as np
from collections import namedtuple, deque

# Pytorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Implementation imports
from ddpg.src.model import Actor, Critic 
from ddpg.src.utils import OUNoise, ReplayBuffer, flatten

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 2 		# 
ACTOR_LR =  5e-4 #5e-3			# actor learning rate
CRITIC_LR = 5e-4 #5e-4		# critic learning rate

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def copy_weights(source_network, target_network):
	"""Copy source network weights to target"""
	for target_param, source_param in zip(target_network.parameters(), source_network.parameters()):
		target_param.data.copy_(source_param.data)


class DDPG_Agent():
	#''' DDPG agent '''
	def __init__(self, state_size, action_size, num_agents, seed, actor_hidden_layers, critic_hidden_layers, use_batch_norm=False):
		super(DDPG_Agent, self).__init__()
		
		self.state_size = state_size
		self.action_size = action_size
		
		self.random_seed = random.seed(seed)
		
		# Actor networks
		self.actor_local = Actor(state_size, action_size, seed, actor_hidden_layers, use_batch_norm).to(device)
		self.actor_target = Actor(state_size, action_size, seed, actor_hidden_layers, use_batch_norm).to(device)
		self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=ACTOR_LR)
		copy_weights(self.actor_local, self.actor_target)
		
		# Critic networks
		self.critic_local = Critic(state_size, action_size, seed, critic_hidden_layers).to(device)
		self.critic_target = Critic(state_size, action_size, seed, critic_hidden_layers).to(device)
		self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=CRITIC_LR)
		copy_weights(self.critic_local, self.critic_target)
		
		# Replay Memory
		self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
		
		# Noise process
		self.noise = OUNoise((num_agents, action_size), seed)
		

		self.t_step = 0

	def step(self, states, actions, rewards, next_states, dones):
		''' Save experience in replay memory, and use random sample from buffer to learn. '''
		for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
			self.memory.add(state, action, reward, next_state, done)

		# update time steps
		self.t_s = (self.t_step + 1) % UPDATE_EVERY
		if self.t_step == 0:
			# time to learn again
			# provided that there are enough
			#if len(shared_memory.shared_buffer) > BATCH_SIZE:
			if len(self.memory) > BATCH_SIZE:
				#experiences = self.memory.sample()
				#experiences = shared_memory.shared_buffer.sample()
				experiences = self.memory.sample()
				self.learn(experiences, GAMMA)

	def act(self, state, noise_weight):
		''' Returns actions for a given state as per current policy '''
		
		# Make current state into a Tensor that can be passed as input to the network
		state = torch.from_numpy(state).float().to(device)

		# Set network in evaluation mode to prevent things like dropout from happening
		self.actor_local.eval()

		# Turn off the autograd engine
		with torch.no_grad():
			# Do a forward pass through the network
			action_values = self.actor_local(state).cpu().data.numpy()

		# Put network back into training mode
		self.actor_local.train()
		
		
		action_values += self.noise.sample() * noise_weight

		return np.clip(action_values, -1, 1)
		
	def reset(self):
		''' Reset the noise in the OU process '''
		self.noise.reset()


	def learn(self, experiences, gamma):
		''' Q_targets = r + γ * critic_target(next_state, actor_target(next_state)) '''

		states, actions, rewards, next_states, dones = experiences

		# ------------------------ Update Critic Network ------------------------ #
		next_actions = self.actor_target(next_states)
		Q_targets_prime = self.critic_target(next_states, next_actions)

		# Compute y_i
		Q_targets = rewards + (gamma * Q_targets_prime * (1 - dones))

		# Compute the critic loss
		Q_expected = self.critic_local(states, actions)
		critic_loss = F.mse_loss(Q_expected, Q_targets)
		# Minimise the loss
		self.critic_optimizer.zero_grad() # Reset the gradients to prevent accumulation
		critic_loss.backward()            # Compute gradients
		torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
		self.critic_optimizer.step()      # Update weights

		# ------------------------ Update Actor Network ------------------------- #
		# Compute the actor loss
		actions_pred = self.actor_local(states)
		actor_loss = -self.critic_local(states, actions_pred).mean()

		# Minimise the loss
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()


		# ------------------------ Update Target Networks ----------------------- #
		self.soft_update(self.critic_local, self.critic_target, TAU)
		self.soft_update(self.actor_local, self.actor_target, TAU)
		
	def soft_update(self, local_model, target_model, tau):
		"""Soft update model parameters.
		θ_target = τ*θ_local + (1 - τ)*θ_target

		Params
		======
			local_model (PyTorch model): weights will be copied from
			target_model (PyTorch model): weights will be copied to
			tau (float): interpolation parameter 
		"""
		for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
			target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class MADDPG_Agent():
	#''' DDPG agent '''
	def __init__(self, state_size, action_size, num_agents, seed, actor_hidden_layers, critic_hidden_layers, use_batch_norm=False):
		super(MADDPG_Agent, self).__init__()

		self.state_size = state_size
		self.action_size = action_size
		self.num_agents = num_agents

		self.random_seed = random.seed(seed)

		self.actor_local = []
		self.actor_target = []
		self.actor_optimizer = []

		self.critic_local = []
		self.critic_target = []
		self.critic_optimizer = []

		for agent in range(num_agents):
			# Actor networks
			self.actor_local.append(Actor(state_size, action_size, seed, actor_hidden_layers, use_batch_norm).to(device))
			self.actor_target.append(Actor(state_size, action_size, seed, actor_hidden_layers, use_batch_norm).to(device))
			copy_weights(self.actor_local[agent], self.actor_target[agent])

			# Critic networks
			# Critic networks see all states and actions
			self.critic_local.append(Critic(state_size*num_agents, action_size*num_agents, seed, critic_hidden_layers).to(device))
			self.critic_target.append(Critic(state_size*num_agents, action_size*num_agents, seed, critic_hidden_layers).to(device))
			copy_weights(self.critic_local[agent], self.critic_target[agent])

			# Optimizers
			self.actor_optimizer.append(optim.Adam(self.actor_local[agent].parameters(), lr=ACTOR_LR))
			self.critic_optimizer.append(optim.Adam(self.critic_local[agent].parameters(), lr=CRITIC_LR))
			

		# Replay Memory
		self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)

		# Noise process
		self.noise = OUNoise(action_size, seed)
		
		self.t_step = 0

	def step(self, states, actions, rewards, next_states, dones):
		''' Save experience in replay memory, and use random sample from buffer to learn. '''
		
		states = np.expand_dims(states, 0)
		actions = np.expand_dims(np.array(actions).reshape(self.num_agents, self.action_size),0)
		rewards = np.expand_dims(np.array(rewards).reshape(self.num_agents, -1),0)
		dones = np.expand_dims(np.array(dones).reshape(self.num_agents, -1),0)
		next_states = np.expand_dims(np.array(next_states).reshape(self.num_agents, -1), 0)
		self.memory.add(states, actions, rewards, next_states, dones)
		
		# update time steps
		self.t_s = (self.t_step + 1) % UPDATE_EVERY
		if self.t_step == 0:
			# time to learn again, provided that there are enough experience tuples in the buffer
			if len(self.memory) > BATCH_SIZE:
				experiences = self.memory.sample()
				self.learn(experiences, GAMMA)

	def act(self, states, noise_weight):
		''' Returns actions for a given state as per current policy '''

		actions = []

		for agent in range(self.num_agents):
			# Make current state into a Tensor that can be passed as input to the network
			state = torch.from_numpy(states[agent]).float().unsqueeze(0).to(device)

			# Set network in evaluation mode to prevent things like dropout from happening
			self.actor_local[agent].eval()

			# Turn off the autograd engine
			with torch.no_grad():
				# Do a forward pass through the network
				action_values = self.actor_local[agent](state).cpu().data.numpy()

			# Put network back into training mode
			self.actor_local[agent].train()

			action_values += self.noise.sample() * noise_weight

			actions.append(np.clip(action_values, -1, 1))

		return actions


	def reset(self):
		''' Reset the noise in the OU process '''
		self.noise.reset()


	def learn(self, experiences, gamma):
		''' Q_targets = r + γ * critic_target(next_state, actor_target(next_state)) '''

		states, actions, rewards, next_states, dones = experiences

		# ------------------------ Update Critic Network ------------------------ #
		
		next_actions = torch.zeros((len(next_states), self.num_agents, self.action_size)).to(device)
		for agent in range(self.num_agents):            
			next_actions[:, agent] = self.actor_target[agent](next_states[:, agent, :].contiguous())
		
		
		for agent in range(self.num_agents):
			# Compute the common 
			Q_targets_prime = self.critic_target[agent](flatten(next_states), flatten(next_actions))
			
			# Compute y_i
			Q_targets = rewards[:,agent,:] + (gamma * Q_targets_prime * (1 - dones[:,agent,:]))

			# Compute the critic loss
			Q_expected = self.critic_local[agent](flatten(states), flatten(actions))
			critic_loss = F.mse_loss(Q_expected, Q_targets)
			critic_loss_value = critic_loss.item()
			# Minimise the loss
			self.critic_optimizer[agent].zero_grad() # Reset the gradients to prevent accumulation
			# Compute gradients
			if agent == self.num_agents - 1:
				critic_loss.backward()
			else:
				critic_loss.backward(retain_graph=True)            
				
			torch.nn.utils.clip_grad_norm_(self.critic_local[agent].parameters(), 1)
			self.critic_optimizer[agent].step()      # Update weights
			
			
		


		# ------------------------ Update Actor Network ------------------------- #
		
		actions_pred = torch.zeros((len(states), self.num_agents, self.action_size)).to(device)
		for agent in range(self.num_agents):            
			actions_pred[:, agent] = self.actor_local[agent](states[:, agent, :].contiguous())
		
		for agent in range(self.num_agents):
			# Compute the actor loss
			actor_loss = -self.critic_local[agent](flatten(states), flatten(actions_pred)).mean()

			# Minimise the loss
			self.actor_optimizer[agent].zero_grad()
			if agent == self.num_agents - 1:
				actor_loss.backward()
			else:
				actor_loss.backward(retain_graph=True)
			self.actor_optimizer[agent].step()


		# ------------------------ Update Target Networks ----------------------- #
		for agent in range(self.num_agents):
			self.soft_update(self.critic_local[agent], self.critic_target[agent], TAU)
			self.soft_update(self.actor_local[agent], self.actor_target[agent], TAU)
			

	def soft_update(self, local_model, target_model, tau):
		"""Soft update model parameters.
		θ_target = τ*θ_local + (1 - τ)*θ_target

		Params
		======
			local_model (PyTorch model): weights will be copied from
			target_model (PyTorch model): weights will be copied to
			tau (float): interpolation parameter 
		"""
		for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
			target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)