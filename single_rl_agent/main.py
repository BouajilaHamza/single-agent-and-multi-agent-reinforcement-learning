import numpy as np
import gymnasium as gym
import mlflow
import logging
import time
import matplotlib.pyplot as plt

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


alpha = 0.1
gamma = 0.99
epsilon = 0.1
episodes = 1000


env = gym.make("FrozenLake-v1", is_slippery=True)
n_states = env.observation_space.n #getting number of states from environment why ? because environment is frozen lake thus it has 16 states
n_actions = env.action_space.n #getting number of actions from environment why ? because environment is frozen lake thus it has 4 actions

Q = np.zeros((n_states, n_actions)) 


mlflow.start_run()
mlflow.log_params({"alpha": alpha, "gamma": gamma, "epsilon": epsilon})

def visualize_episode(env, Q, delay=0.5):
    """
    Visualize a single episode using the learned Q-table
    """
    state = env.reset(seed=42)
    env.render()
    total_reward = 0
    steps = []
    
    while True:
        action = np.argmax(Q[state, :])
        next_state, reward, done, _ = env.step(action)
        steps.append((state, action, reward))
        env.render()
        time.sleep(delay)
        total_reward += reward
        state = next_state
        if done:
            break
    
    return total_reward, steps

def plot_training_progress(rewards):
    """
    Plot the training progress
    """
    plt.figure(figsize=(12, 6))
    plt.plot(rewards)
    plt.title('Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.savefig('training_progress.png')
    plt.close()

def evaluate(Q, n_episodes=100):
    """
    Evaluate the agent's performance over multiple episodes
    """
    logger.info(f"Starting evaluation with {n_episodes} episodes")
    total_reward = 0
    episode_rewards = []
    
    for i in range(n_episodes):
        s, _ = env.reset()
        done = False
        episode_reward = 0
        step_count = 0
        
        while not done:
            a = np.argmax(Q[s])
            s, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            logger.info(f"Step {step_count}: State {s}, Action {a}, Reward {r}")
            episode_reward += r
            step_count += 1
            
        total_reward += episode_reward
        episode_rewards.append(episode_reward)
        logger.info(f"Episode {i+1}/{n_episodes} - Reward: {episode_reward}, Steps: {step_count}")
    logger.info(f"total reward: {total_reward} / {n_episodes} episodes  ")
    avg_reward = total_reward / n_episodes
    logger.info(f"Evaluation complete. Average reward over {n_episodes} episodes: {avg_reward:.3f}")
    logger.info(f"Best episode reward: {max(episode_rewards)}")
    logger.info(f"Worst episode reward: {min(episode_rewards)}")
    
    mlflow.log_metric("avg_eval_reward", avg_reward)
    mlflow.log_metric("best_eval_reward", max(episode_rewards))
    mlflow.log_metric("worst_eval_reward", min(episode_rewards))
    
    return avg_reward

def choose_action(s):
    if np.random.rand() < epsilon:
        return np.random.choice(n_actions)
    else:
        return np.argmax(Q[s])

# Training loop
rewards = []
for ep in range(episodes):
    s, _ = env.reset()
    done = False
    episode_reward = 0
    
    while not done:
        a = choose_action(s)
        s_next, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
        Q[s, a] += alpha * (r + gamma * np.max(Q[s_next]) - Q[s, a])
        episode_reward += r
        s = s_next
    
    mlflow.log_metric("episode_reward", episode_reward, step=ep)
    rewards.append(episode_reward)

    if ep % 100 == 0:
        avg_reward = evaluate(Q)
        logger.info(f"Episode {ep}: Average reward over 100 episodes = {avg_reward:.4f}")
        mlflow.log_metric("average_reward", avg_reward, step=ep)
        
    if ep % 1000 == 0:
        logger.info(f"Completed {ep} episodes")
        avg_reward = evaluate(Q)
        mlflow.log_metric("avg_reward", avg_reward, step=ep)


np.save("Q_table.npy", Q)
mlflow.log_artifact("Q_table.npy")
mlflow.end_run()

best_actions = np.argmax(Q, axis=1).reshape(4,4)
# visualize heatmap + arrows...

# To play with trained policy:
Q = np.load("Q_table.npy")
state, _ = env.reset()
done = False
while not done:
    action = np.argmax(Q[state])
    state, r, done, _, _ = env.step(action)
    env.render()

