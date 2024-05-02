# Author: Vibhakar Mohta (vmohta@cs.cmu.edu)

import numpy as np
import matplotlib.pyplot as plt
from environment import GridWorld
from agents import RandomAgent, QAgent
import os

# Define Hyperparameters
MAX_STEPS_PER_EPISODE = 100
BENCHMARK_TRIALS = 500
N_TRAINING_TRIALS = 1000
ENV_RHO = 0.01
Q_ALPHA = 0.1
Q_GAMMA = 0.99
Q_EPSILON = 0.1        
        
def train_q_agent(environment: GridWorld, agent: QAgent, trials=1000, max_steps_per_episode=100):
    """
    Train the agent in the environment for a number of trials
    """
    # TODO: Implement this function
    reward_per_episode = []
    for _ in range(trials):
        state = environment.reset()
        cumulative_reward = 0
        step = 0
        while step < max_steps_per_episode:
            # Train loop for Q agent
            
            if done:
                break
                
        reward_per_episode.append(cumulative_reward)
    return reward_per_episode

def visualize_agent_run(environment: GridWorld, agent, max_steps_per_episode=100, title="", num_visuals=3):
    """
    Save a video of the agent running in the environment
    """
    # Make directory visualizations if it does not exist
    if not os.path.exists('visualizations'):
        os.makedirs('visualizations')
    
    # Visualize the agent running in the environment
    for i in range(num_visuals):
        state = environment.reset()
        environment.clear_display()
        step = 0
        cumulative_reward = 0
        while step < max_steps_per_episode:
            environment.render_grid(title=f"Agent {title}, Step {step}, Total Reward: {cumulative_reward}")
            action = agent.get_action(state, explore=False)
            state, reward, done = environment.step(action)
            cumulative_reward += reward        
            step += 1
            
            if done:
                break
        environment.render_grid(title=f"Agent {title}, Step {step}, Total Reward: {cumulative_reward}")
        environment.save_display(f'visualizations/{title}_{i}.mp4')
            
        print("Finished after", step, "steps with total reward of", cumulative_reward)

def benchmark_performance(environment: GridWorld, agent, trials=100, max_steps_per_episode=100):
    """
    Benchmark the agent in the environment for a number of trials
    """
    reward_per_episode = [] 
    for _ in range(trials): 
        environment.reset()
        cumulative_reward = 0
        step = 0
        while step < max_steps_per_episode:
            action = agent.get_action(environment.current_location, explore=False) # Greedy action
            _, reward, done = environment.step(action)
                            
            cumulative_reward += reward
            step += 1
            
            if done:
                break
        reward_per_episode.append(cumulative_reward) 
        
    return reward_per_episode

if __name__ == '__main__':
    # Seed random number generator
    np.random.seed(...) # TODO: set your own random seed
    env = GridWorld(rho=ENV_RHO)
    
    # Benchmark random agent
    print("Benchmarking Random Agent for {} trials".format(BENCHMARK_TRIALS))
    random_agent = RandomAgent()
    rewards_random = benchmark_performance(env, agent=random_agent, trials=BENCHMARK_TRIALS,
                                            max_steps_per_episode=MAX_STEPS_PER_EPISODE)
    print("Average reward random agent:", np.mean(rewards_random))
    
    print("\nVisualizing Random Agent")
    visualize_agent_run(env, random_agent, title="random_agent")
    
    # Train Q agent
    print("\nTraining Q Agent for {} trials".format(N_TRAINING_TRIALS))
    agentQ = QAgent(env, alpha=Q_ALPHA, gamma=Q_GAMMA, epsilon=Q_EPSILON)
    train_rewards = train_q_agent(env, agent=agentQ, trials=N_TRAINING_TRIALS,
                                    max_steps_per_episode=MAX_STEPS_PER_EPISODE)
    
    plt.figure(figsize=(10, 5))
    plt.plot(train_rewards)
    plt.xlabel('Trial')
    plt.ylabel('Reward')
    plt.title('Q-Learning: Rewards vs Trials')
    plt.savefig('q_learning_training.png')
    
    # Benchmark Q agent
    print("\nBenchmarking Q Agent for {} trials".format(BENCHMARK_TRIALS))
    rewards_q = benchmark_performance(env, agent=agentQ, trials=BENCHMARK_TRIALS,
                                      max_steps_per_episode=MAX_STEPS_PER_EPISODE)
    print("Average reward Q agent:", np.mean(rewards_q))
    
    print("\nVisualizing Q Agent")
    visualize_agent_run(env, agentQ, title="q_agent")
    
    # visualize Q values
    agentQ.visualize_q_values()