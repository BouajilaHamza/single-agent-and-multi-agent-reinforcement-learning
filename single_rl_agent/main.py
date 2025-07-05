import numpy as np
import gymnasium as gym
import mlflow

# Config
alpha = 0.1
gamma = 0.99
epsilon = 0.1
episodes = 1000

# Environment
env = gym.make("FrozenLake-v1", is_slippery=True)
n_states = env.observation_space.n
n_actions = env.action_space.n

Q = np.zeros((n_states, n_actions))

# MLflow setup
mlflow.start_run()
mlflow.log_params({"alpha": alpha, "gamma": gamma, "epsilon": epsilon})

def choose_action(state):
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    return np.argmax(Q[state])

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
    while not done:
        a = choose_action(s)
        s_next, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
        Q[s, a] += alpha * (r + gamma * np.max(Q[s_next]) - Q[s, a])
        s = s_next

    if ep % 100 == 0:
        avg_reward = evaluate(Q)
        mlflow.log_metric("avg_reward", avg_reward, step=ep)

# Save final Q-table
np.save("Q_table.npy", Q)
mlflow.log_artifact("Q_table.npy")
mlflow.end_run()
