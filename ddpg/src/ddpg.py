from unityagents import UnityEnvironment
import numpy as np
from collections import deque
import torch

NOISE_DECAY = 0.98

def ddpg_show(agent, env, brain_name, num_agents, actor_model_names, multi_agent=False):

	if multi_agent:
		for i in range(num_agents):
			agent.actor_local[i].load_state_dict(torch.load(actor_model_names[i]))
			agent.actor_local[i].eval()
	else:
		agent.actor_local.load_state_dict(torch.load(actor_model_names))
		agent.actor_local.eval()
	

	env_info = env.reset(train_mode=False)[brain_name] # reset the environment
	states = env_info.vector_observations            # get the current state
	score = np.zeros(num_agents)                                      # initialize the score
	
	while True:
		actions = agent.act(states, 0)                   # select an action
		env_info = env.step(actions)[brain_name]        # send the action to the environment
		next_states = env_info.vector_observations   # get the next state
		rewards = env_info.rewards                  # get the reward
		dones = env_info.local_done                  # see if episode has finished
		score += rewards                                # update the score
		states = next_states                             # roll over the state to next time step
		if np.any(dones):                                       # exit loop if episode finished
			break
		
	print("Mean Score (for all agents): {}".format(np.mean(score)))
	print("Score for individual agents:")
	print(score)

def ddpg_train(agent, env, brain_name, num_agents, actor_model_name, critic_model_name, target_reward, min_noise_coef = 0.05, n_episodes=3000, max_steps=1000, multi_agent=False):
	
	# Keep track of scores
	scores = []
	scores_window_100 = deque(maxlen=100)
	scores_window_20 = deque(maxlen=20)
	
	solved = False

	# Noise weight for training, this will slowly decay to min_noise_coef
	noise_weight = 1.0

	for episode in range(1, n_episodes+1):
		
		env_info = env.reset(train_mode=True)[brain_name]
		states = env_info.vector_observations
		score = np.zeros(num_agents)
		# Reset the noise in the agents
		agent.reset()
		
		for t in range(max_steps):
			actions = agent.act(states, noise_weight)
			env_info = env.step(actions)[brain_name]
			next_states = env_info.vector_observations
			rewards = env_info.rewards
			dones = env_info.local_done
			score += rewards
			agent.step(states, actions, rewards, next_states, dones)

			
			states = next_states
			if np.any(dones):
				break
				
		scores_window_100.append(np.mean(score))
		scores_window_20.append(score)
		scores.append(np.mean(score))        
				
		if episode % 10 == 0:
			print('\rEpisode {}\tAverage Score: {:.2f}\tAverage last 100 episodes: {:.2f}'.format(episode, np.mean(scores_window_20), np.mean(scores_window_100)))
		
		# Save models every 100th episodes
		if episode % 100 == 0:
			if multi_agent:
				for i in range(num_agents):
					torch.save(agent.actor_local[i].state_dict(), 'ddpg/results/' + actor_model_name + '_agent_' + str(i) + '_epi_' + str(episode) + '.pth')
					torch.save(agent.critic_local[i].state_dict(), 'ddpg/results/' + critic_model_name + '_agent_' + str(i) + '_epi_' + str(episode) + '.pth')
			else:
				torch.save(agent.actor_local.state_dict(), 'ddpg/results/' + actor_model_name + '_epi_' + str(episode) + '.pth')
				torch.save(agent.critic_local.state_dict(), 'ddpg/results/' + critic_model_name + '_epi_' + str(episode) + '.pth')
		
		# Decay noise weight
		if np.mean(scores_window_100) > 0.1 * target_reward:
			noise_weight = max(min_noise_coef, noise_weight*NOISE_DECAY)
		
		
		# Check if environment has been solved
		if np.mean(scores_window_100)>= target_reward and not solved:
			# Agent has reached target average score.
			print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode-100, np.mean(scores_window_100)))
			# Save models for reaching target reward
			if multi_agent:
				for i in range(num_agents):
					torch.save(agent.actor_local[i].state_dict(), 'ddpg/results/' + actor_model_name + '_agent_' + str(i) + '_solved_env.pth')
					torch.save(agent.critic_local[i].state_dict(), 'ddpg/results/' + critic_model_name + '_agent_' + str(i) + '_solved_env.pth')
			else:
				torch.save(agent.actor_local.state_dict(), 'ddpg/results/' + actor_model_name + '_solved_env.pth')
				torch.save(agent.critic_local.state_dict(), 'ddpg/results/' + critic_model_name + '_solved_env.pth')
			solved = True
	
	# Save the models for end of training
	if multi_agent:
		for i in range(num_agents):
			torch.save(agent.actor_local[i].state_dict(), 'ddpg/results/' + actor_model_name + '_agent_' + str(i) + '_end_train.pth')
			torch.save(agent.critic_local[i].state_dict(), 'ddpg/results/' + critic_model_name + '_agent_' + str(i) + '_end_train.pth')
	else:
		torch.save(agent.actor_local.state_dict(), 'ddpg/results/' + actor_model_name + '_end_train.pth')
		torch.save(agent.critic_local.state_dict(), 'ddpg/results/' + critic_model_name + '_end_train.pth')
	
	print("Finished training")
	
	return scores