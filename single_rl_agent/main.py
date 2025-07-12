import numpy as np
import gymnasium as gym
import mlflow
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)

# Config
alpha = 0.01 #learning rate
gamma = 0.99 #discount factor 
epsilon = 0.05 #exploration rate: probability of taking a random action
episodes = 100000 #number of episodes : episode is a sequence of states, actions, and rewards

# Environment
env = gym.make("FrozenLake-v1", is_slippery=True)
n_states = env.observation_space.n #getting number of states from environment why ? because environment is frozen lake thus it has 16 states
n_actions = env.action_space.n #getting number of actions from environment why ? because environment is frozen lake thus it has 4 actions

Q = np.zeros((n_states, n_actions)) 

# MLflow setup
mlflow.start_run()
mlflow.log_params({"alpha": alpha, "gamma": gamma, "epsilon": epsilon})

def choose_action(state):
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    return np.argmax(Q[state]) #returns the action with the highest Q-value for the given state

def evaluate(Q, n_episodes=100):
    total_reward = 0
    for _ in range(n_episodes):
        s, _ = env.reset()
        done = False
        while not done:
            a = np.argmax(Q[s])
            s, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            total_reward += r
    return total_reward / n_episodes

# Training
for ep in range(episodes):
    s, _ = env.reset()
    done = False
    episode_reward = 0
    
    while not done:
        a = choose_action(s)
        s_next, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
        Q[s, a] += alpha * (r + gamma * np.max(Q[s_next]) - Q[s, a])
        s = s_next
        episode_reward += r

    if ep % 100 == 0:
        avg_reward = evaluate(Q)
        logger.info(f"Episode {ep}: Average reward over 100 episodes = {avg_reward:.4f}")
        mlflow.log_metric("average_reward", avg_reward, step=ep)
        
    if ep % 1000 == 0:
        logger.info(f"Completed {ep} episodes")
        avg_reward = evaluate(Q)
        mlflow.log_metric("avg_reward", avg_reward, step=ep)

# Save final Q-table
np.save("Q_table.npy", Q)
mlflow.log_artifact("Q_table.npy")
mlflow.end_run()
