**main algorithms** used in **single-agent reinforcement learning**, focusing on:

* 📌 **Definitions**
* 🧠 **Intuition**
* ⚙️ **How they work**
* 🧪 **How to test and evaluate**
* 🔍 **How to interpret results**

---

# 🧠 Single-Agent RL Algorithms: Intuition, Explanation, Evaluation

---

## 🔸 1. **Q-Learning** (Off-Policy Value-Based)

### 📌 What is it?

Q-Learning is a **model-free**, **off-policy** algorithm that learns the **optimal action-value function** $Q^*(s, a)$ directly.

### 🧠 Intuition:

* Learn how good each **action** is in each **state**.
* Pick actions that lead to **maximum future reward**.

> Imagine playing a video game repeatedly and remembering which moves got you the most points — that's Q-learning.

### ⚙️ Update Rule:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s,a) \right]
$$

Where:

* $\alpha$: learning rate
* $\gamma$: discount factor
* $s$: current state
* $a$: action taken
* $r$: reward received
* $s'$: next state

### 🔄 Process:

1. Initialize Q-table (or Q-network)
2. Choose action (e.g., ε-greedy)
3. Take action → observe reward & next state
4. Update Q-value
5. Repeat

### ✅ Strengths:

* Simple and effective
* Can learn optimal policies

### ❌ Weaknesses:

* Doesn’t scale well to large state spaces
* Needs full state-action enumeration

---

## 🔸 2. **SARSA** (On-Policy Value-Based)

### 📌 What is it?

SARSA (State-Action-Reward-State-Action) is similar to Q-Learning but learns the **value of the current policy**, not the optimal one.

### 🧠 Intuition:

* Learns action values **based on what the agent actually does**, not what it could have done.

### ⚙️ Update Rule:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma Q(s', a') - Q(s,a) \right]
$$

* Uses **actual next action** $a'$ instead of max over all actions.

### ✅ SARSA vs Q-Learning:

* SARSA is **on-policy**: more conservative (safer behavior)
* Q-learning is **off-policy**: more aggressive (optimal behavior)

---

## 🔸 3. **Deep Q-Network (DQN)**

### 📌 What is it?

DQN extends Q-Learning using a **neural network** to approximate the Q-function.

### 🧠 Intuition:

* Use deep learning to scale Q-learning to **complex problems** (e.g., games with images as input).

### ⚙️ Key Components:

* **Q-network**: Approximates $Q(s, a; \theta)$
* **Experience replay**: Stores and samples transitions to break correlation
* **Target network**: Stable target for learning

### 🔄 Training Loop:

1. Collect experiences $(s, a, r, s')$
2. Sample mini-batch
3. Compute target $y = r + \gamma \max_{a'} Q_{\text{target}}(s', a')$
4. Minimize loss: $\mathcal{L} = (Q(s, a) - y)^2$
5. Update Q-network

### ✅ Strengths:

* Works on high-dimensional problems (e.g., Atari games)
* Generalizes across similar states

### ❌ Weaknesses:

* Requires lots of data
* Training can be unstable

---

## 🔸 4. **Policy Gradient Methods**

### 📌 What is it?

Instead of learning value functions, these directly learn the **policy** $\pi(a|s)$ by optimizing performance using gradient ascent.

### 🧠 Intuition:

* Treat policy as a **black box** and adjust it to get **better rewards**.

### ⚙️ Objective:

$$
J(\theta) = \mathbb{E}_\pi \left[ \sum_{t=0}^\infty \gamma^t r_t \right]
$$

Use gradient ascent:

$$
\nabla_\theta J(\theta) = \mathbb{E}_\pi \left[ \nabla_\theta \log \pi_\theta(a|s) \cdot Q^\pi(s,a) \right]
$$

### ✅ Strengths:

* Works well with continuous action spaces
* Good for stochastic policies

### ❌ Weaknesses:

* High variance in gradients
* Less sample-efficient than value-based methods

---

## 🔸 5. Actor-Critic Methods

### 📌 What is it?

Combines:

* **Actor**: learns policy $\pi(a|s)$
* **Critic**: learns value function $V(s)$ or $Q(s,a)$

### 🧠 Intuition:

* Actor makes decisions, critic judges them and tells the actor how to improve.

### ⚙️ Examples:

* A2C (Advantage Actor-Critic)
* PPO (Proximal Policy Optimization)

---

## 🔍 How to Test, Evaluate, and Interpret RL Agents

### ✅ 1. **Evaluation Metrics**

| Metric                | Description                             |
| --------------------- | --------------------------------------- |
| **Average Return**    | Mean total reward per episode           |
| **Episode Length**    | Time steps before termination           |
| **Convergence Speed** | How fast learning happens               |
| **Stability**         | How consistent performance is over time |

---

### ✅ 2. **Testing Policy**

* Disable exploration (ε = 0 or no randomness)
* Run for several episodes
* Log average return and standard deviation

---

### ✅ 3. **Visualize Learning**

* **Learning curve:** Plot average reward per episode
* **Policy visualization:** For discrete environments, visualize state → action mapping
* **Value function heatmaps:** Show how values evolve over time

---

### ✅ 4. **Interpret Results**

| Observation           | Interpretation                                  |
| --------------------- | ----------------------------------------------- |
| Steady rise in return | Agent is learning well                          |
| High variance         | Policy is unstable or exploration is too high   |
| Early plateau         | Learning rate may be too low or policy is stuck |
| Oscillation           | Overfitting or poor hyperparameters             |

---

## 🧪 Example Evaluation Code (with Gym)

```python
def evaluate_policy(env, policy, episodes=10):
    total_rewards = []
    for _ in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = policy(state)  # Use learned policy (e.g., greedy Q)
            state, reward, done, _ = env.step(action)
            total_reward += reward
        total_rewards.append(total_reward)
    return np.mean(total_rewards), np.std(total_rewards)
```

---

## 📚 Summary

| Algorithm       | Learns   | Type       | Strengths                    |                             |
| --------------- | -------- | ---------- | ---------------------------- | --------------------------- |
| Q-Learning      | $Q(s,a)$ | Off-policy | Simple, converges to optimal |                             |
| SARSA           | $Q(s,a)$ | On-policy  | Safer, more stable           |                             |
| DQN             | $Q(s,a)$ | Off-policy | Handles large state spaces   |                             |
| Policy Gradient | ( \pi(a  | s) )       | On-policy                    | Works in continuous actions |
| Actor-Critic    | $\pi, V$ | Mixed      | Balanced learning            |                             |

---

Would you like:

1. To implement Q-learning or DQN in `CartPole` or `FrozenLake`?
2. Help with tuning hyperparameters and debugging?
3. Explore advanced evaluation like policy entropy, success rate, etc.?
