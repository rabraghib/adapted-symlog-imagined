# Results Analysis

This document provides a detailed breakdown of the 6 generated graphs from our experiment suite, explaining how they connect to the core hypothesis of the project.

---

## Group 1: CartPole-v1 (The Simple Environment)

### 1. `CartPole-v1_learning_curves.png`
* **What it shows**: The evaluation return (how long the pole stays balanced, max 500) over 100,000 training steps.
* **Analysis**: 
  * **Model-Free (Green)**: The standard baseline. It learns steadily but relatively slowly, reaching about 350 by the end.
  * **Imagination without Symlog (Blue)**: It is much more sample-efficient. Because it trains on imagined rollouts, it learns much faster in the beginning, reaching ~400 at 60k steps. But notice how it plateaus early—it struggles to refine its policy further.
  * **Imagination + Symlog (Orange)**: It starts a bit slower but eventually overtakes both, reaching the highest final return (~430). 

### 2. `CartPole-v1_loss_comparison.png`
* **What it shows**: The training loss of the Critic network (Value network).
* **Analysis**: This is a key result! The blue line (Imagination without Symlog) completely explodes at the end of training, reaching `5e11`. When an agent plans using imagined future states, small errors in value estimation compound rapidly (value explosion). The orange line (with Symlog) perfectly suppresses this, staying completely flat near 0. Symlog mathematically prevents the values from spiraling out of control.

### 3. `CartPole-v1_world_model_error.png`
* **What it shows**: How accurately the World Model predicts the next state (Dynamics) and the reward (Reward).
* **Analysis**: CartPole is a very simple physics environment and always gives a +1 reward for surviving. Therefore, both the blue and orange lines drop to near zero almost instantly. The world model easily learns this environment regardless of whether Symlog is used.

---

## Group 2: LunarLander-v3 (The Hard Environment)

LunarLander is much harder because the rewards vary wildly (e.g., -100 for crashing, +100 for landing, small negative rewards for using fuel).

### 4. `LunarLander-v3_learning_curves.png`
* **What it shows**: Evaluation return over 300,000 steps. A score of +200 is considered "solved", but with simple MLPs, agents often get stuck in local optima (like hovering safely but never landing).
* **Analysis**:
  * **Model-Free (Green)** & **Imagination + Symlog (Orange)**: Both agents learn to avoid crashing and converge to a stable local optimum around `-140`. They perform identically at the end.
  * **Imagination without Symlog (Blue)**: Total catastrophic failure. It completely collapses to `-1000` very early on and never recovers.

### 5. `LunarLander-v3_loss_comparison.png`
* **What it shows**: The Critic loss.
* **Analysis**: This graph explains *why* the blue agent failed. Around 100,000 steps, the blue line's loss explodes to **`1e15`** (one quadrillion). The neural network weights were completely destroyed by exploding gradients. Because LunarLander has extreme negative rewards for crashing, the imagination rollout compounds those negative values into infinity. The orange agent (with Symlog) perfectly compresses these values, keeping the loss completely flat and stable, allowing the agent to survive.

### 6. `LunarLander-v3_world_model_error.png`
* **What it shows**: World Model prediction errors.
* **Analysis**: Look at the **Reward Prediction Error** on the right. The blue line (no Symlog) is constantly spiking massively between 0 and 450 MSE. It simply cannot figure out the wild swings in LunarLander's reward function. Meanwhile, the orange line (Symlog) is pinned perfectly flat at zero. By predicting rewards in the compressed logarithmic space, the world model easily masters the reward function without being destabilized by outliers.

---

## Conclusion

The graphs perfectly validate the core hypothesis of the project:
1. **Imagination (Dyna-style planning)** makes learning more sample-efficient (shown in CartPole).
2. However, imagination suffers from **Value Explosion** and **Reward scaling instability**, especially in environments with varied rewards like LunarLander (shown by the exploding loss and collapsed learning curve).
3. **Symlog transformations** completely solve this instability, preventing value explosion and allowing the imagination agent to match or beat the model-free baseline reliably.
