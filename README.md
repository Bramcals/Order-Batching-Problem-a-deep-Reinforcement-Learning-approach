# Order-Batching-Problem-a-deep-Reinforcement-Learning-approach
Archive code of Deep Reinforcement Learning approach used by Bram Cals. Code was developed for the graduation research, as a final project within the master program Operations Management and Logistics at the Eindhoven University of Technology. 

RL_agent_SMARTPICK_FinalCleanUp located in "environments, is the main model that can be trained, retrained and tested with train.py, retrain.py and test.py respectively.

The RL_agent_SMARTPICK_FinalCleanUp interacts with the simulation model which is written in Automod. Code of the automod file is found in the "Simulation Model (automod)" folder

The training parameters for the PPO algorithm can be specified in the configure file. Here also the tensorboard scalars can be specified. 

Four trained models are included. Each with their own "config file", trained weights&policy parameters and the event loggings for tensorboard. 
