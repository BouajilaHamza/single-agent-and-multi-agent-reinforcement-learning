# Study single-agent-and-multi-agent-reinforcement-learning


## 🔹 What is Reinforcement Learning (RL)?

Reinforcement learning is a type of machine learning where an **agent** learns to make decisions by interacting with an **environment**, aiming to **maximize cumulative reward** over time.

---

## 🔸 Single-Agent Reinforcement Learning

### ✅ Definition:

In **single-agent RL**, there is **one agent** interacting with a stationary environment. The environment’s dynamics do not change due to other agents — it is passive, only reacting to this single agent.

### 📌 Formalized as a **Markov Decision Process (MDP)**:

* **States (S):** Environment situations
* **Actions (A):** Choices the agent can make
* **Transition function (T):** Probability of reaching new state given current state and action
* **Reward function (R):** Feedback the agent gets for its actions
* **Policy (π):** The strategy used by the agent to choose actions

### 🧠 Goal:

Find a policy π that maximizes the expected **return** (i.e., total reward over time).

### 🛠️ Example Algorithms:

* Q-Learning
* SARSA
* Deep Q-Networks (DQN)
* Policy Gradient methods
* Proximal Policy Optimization (PPO)

### 🧩 Applications:

* Game playing (e.g., Atari games, Chess)
* Robotics (e.g., navigating a robot in a maze)
* Inventory management

---

## 🔸 Multi-Agent Reinforcement Learning (MARL)

### ✅ Definition:

In **multi-agent RL**, **multiple agents** interact in a shared environment. These agents can be **cooperative**, **competitive**, or both (**mixed**).

### 📌 Formalized as:

* **Stochastic Game** (also called a **Markov Game**) — an extension of MDP to multiple agents.
* Each agent has:

  * Its own **state observations**
  * Its own **actions**
  * Its own **reward function**
  * May share or have distinct goals

### 🧠 Challenges:

* **Non-stationarity:** From each agent's perspective, the environment includes other agents whose strategies may change.
* **Partial observability:** Each agent may have limited information.
* **Credit assignment:** Hard to determine which agent’s actions led to a collective outcome.
* **Coordination and Communication:** Especially in cooperative settings.

### 🛠️ Example Algorithms:

* Independent Q-Learning
* Multi-Agent Deep Deterministic Policy Gradient (MADDPG)
* QMIX (for cooperative agents)
* Counterfactual Multi-Agent Policy Gradients (COMA)
* Self-Play (used in AlphaGo, AlphaStar)

### 🧩 Applications:

* Autonomous vehicles in traffic
* Multi-robot systems
* Distributed sensor networks
* Real-time strategy games (e.g., StarCraft)
* Economics and auctions

---

## 🔍 Comparison Table

| Feature                | Single-Agent RL             | Multi-Agent RL                                           |
| ---------------------- | --------------------------- | -------------------------------------------------------- |
| # of Agents            | 1                           | 2 or more                                                |
| Environment            | Stationary                  | Non-stationary (due to other agents)                     |
| Interaction Complexity | Simpler                     | Higher (strategic interactions, cooperation/competition) |
| Learning Objective     | Maximize own reward         | Maximize individual and/or joint reward                  |
| Example Use Case       | Robot arm learning to grasp | Multiple robots moving in coordination                   |

---

## 📚 Further Reading

* Sutton & Barto (2018): *Reinforcement Learning: An Introduction*
* *Multi-Agent Reinforcement Learning: A Selective Overview* — Shoham et al.
* OpenAI MARL projects (e.g., hide-and-seek)
* PettingZoo & SuperSuit: Python libraries for MARL

---

Would you like a diagram or code examples to visualize the differences or get started with implementation?
