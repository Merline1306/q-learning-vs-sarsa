# Q-Learning vs. SARSA: A Comparative Study of Reinforcement Learning

## 1. Project Overview

How can an agent learn to complete a task without being explicitly programmed with every step?

This project explores that question using **Reinforcement Learning (RL)**, a branch of machine learning in which an agent learns through interaction with an environment. Instead of receiving the correct answer for every situation, the agent tries different actions, receives rewards or penalties, and gradually learns which actions lead to better outcomes.

The project implements and compares two reinforcement learning algorithms, **Q-Learning and SARSA**, using the Taxi-v4 environment. Both algorithms learn how to navigate a taxi, pick up a passenger, and deliver the passenger to the correct destination.

The goal is to understand how these algorithms learn, how their performance differs, and how effectively they solve the same task.

## 2. What Is Reinforcement Learning?

Reinforcement Learning is a learning approach based on trial and error.

Imagine teaching a taxi driver to pick up a passenger and drop them at the correct location. Initially, the driver does not know the best route. By trying different actions and receiving feedback, the driver gradually learns a better strategy.

In Reinforcement Learning, this process involves four important elements:

- **Agent:** The learner or decision-maker. In this project, the agent is the taxi.
- **Environment:** The world in which the agent operates. Here, it is the Taxi-v4 simulation.
- **Action:** A decision made by the agent, such as moving north or picking up a passenger.
- **Reward:** Feedback received after an action. Good outcomes receive positive rewards, while incorrect or inefficient actions can receive penalties.

The agent aims to learn a **policy**, which is a strategy for choosing actions in different situations to maximize the total reward over time.

## 3. Understanding the Taxi-v4 Environment

The Taxi-v4 environment is a grid-based simulation provided by Gymnasium. The taxi must collect a passenger from a designated location and transport them to a specified destination.

The environment includes:

- **500 possible states**, representing combinations of the taxi's position, passenger location or status, and destination.
- **6 possible actions:** South, North, East, West, Pickup, and Dropoff.
- A reward system that encourages successful passenger delivery and penalizes incorrect or inefficient behavior.
- Episode termination when the task is completed or the episode reaches its step limit.

The agent begins without a trained policy. Through repeated interactions with the environment, it learns which actions are useful in different states.

This environment is suitable for comparing reinforcement learning algorithms because both algorithms can solve exactly the same task under controlled conditions.

## 4. What Are Q-Learning and SARSA?

Both Q-Learning and SARSA are **model-free, temporal-difference reinforcement learning algorithms**. They learn from experience without requiring a pre-programmed optimal route.

They maintain a Q-table that estimates the expected future reward of taking a particular action in a particular state.

### Q-Learning

Q-Learning is an **off-policy** algorithm. It updates its estimate using the highest estimated action value available in the next state, regardless of which next action the agent actually selects.

In simple terms, it learns using the value of the best estimated future action.

### SARSA

SARSA stands for State–Action–Reward–State–Action. It is an **on-policy** algorithm that updates its estimate using the next action actually selected by the current behavior policy.

In simple terms, it learns using the value of the action the agent is currently following, including during exploration.

### Key Difference

The main difference is how the algorithms estimate future rewards:

- **Q-Learning:** Uses the maximum estimated action value in the next state.
- **SARSA:** Uses the estimated value of the next action actually selected.

Because their update rules differ, the algorithms may learn at different speeds and respond differently to exploration and hyperparameter settings.

## 5. Project Objectives

- Implement Q-Learning and SARSA using tabular methods.
- Train both algorithms in the Taxi-v4 environment.
- Compare their learning behavior and final greedy evaluation performance.
- Evaluate rewards, steps taken, success rates, training penalties, and convergence speed.
- Investigate how learning-rate and exploration-decay settings affect performance.
- Compare learned action values and policies against an optimal-value reference computed using value iteration.
- Visualize learning curves and policy behavior to support the analysis.

## 6. Technologies Used

- **Python:** Implementation of the algorithms and experiment workflow.
- **Gymnasium:** Provides the Taxi-v4 reinforcement learning environment.
- **NumPy:** Numerical calculations and Q-table operations.
- **Pandas:** Organizing experimental results and exporting CSV files.
- **Matplotlib:** Visualizing learning curves and experiment results.
- **SciPy:** Statistical comparison of experimental results.

## 7. Methodology

The project follows these steps:

1. **Environment setup:** Initialize the Taxi-v4 environment and identify its states and actions.
2. **Q-table initialization:** Create a table of action-value estimates for each state-action combination.
3. **Action selection:** Use an epsilon-greedy strategy to balance exploration and exploitation.
4. **Training:** Allow each agent to interact with the environment and update its Q-table after each action.
5. **Policy evaluation:** Test the learned policies by selecting the highest-valued action in each state.
6. **Multi-seed experiments:** Repeat training using five random seeds to compare results across runs.
7. **Sensitivity analysis:** Investigate the effects of changing the learning rate and epsilon-decay settings.
8. **Optimal-value comparison:** Compute reference action values using value iteration and compare them with the learned estimates.
9. **Visualization and analysis:** Generate learning curves, policy visualizations, and summary tables.

### Important Hyperparameters

| Parameter | Value | Purpose |
|---|---:|---|
| Learning rate (alpha) | 0.1 | Controls how much new information changes the Q-value |
| Discount factor (gamma) | 0.99 | Determines the importance of future rewards |
| Initial epsilon | 1.0 | Starts with a high probability of exploration |
| Minimum epsilon | 0.01 | Maintains a small amount of exploration |
| Epsilon decay | 0.995 | Gradually reduces exploration |
| Training episodes | 5,000 | Number of episodes used in the main training configuration |

## 8. Evaluation Metrics

The algorithms are compared using several measures:

- **Greedy evaluation reward:** Average reward obtained when following the learned policy without intentional random exploration.
- **Average steps:** Number of steps required to complete an episode.
- **Success rate:** Percentage of evaluation episodes in which the passenger is delivered successfully.
- **Convergence speed:** Number of training episodes needed for the moving-average reward to reach the selected threshold.
- **Training reward:** Average reward collected throughout training.
- **Training penalties:** Number of penalty events during training.
- **Optimal-action agreement:** Percentage of visited states in which the learned greedy action is optimal according to the value-iteration reference.

The main comparison uses five random seeds, allowing performance variability across runs to be examined.

## 9. Results and Analysis

Under the main experimental configuration, both algorithms learned effective policies for the Taxi-v4 task.

| Evaluation Metric | Q-Learning | SARSA |
|---|---:|---:|
| Mean greedy evaluation reward | 7.63 ± 0.00 | 7.55 ± 0.14 |
| Mean greedy evaluation steps | 13.37 ± 0.00 | 13.45 ± 0.14 |
| Greedy evaluation success rate | 100% | 100% |
| Episodes to reach average reward ≥ 0 | 1,077.60 ± 23.06 | 1,182.20 ± 100.70 |
| Average reward throughout training | -28.29 ± 0.28 | -29.36 ± 0.19 |

*Note: Values are mean ± standard deviation across five random seeds. Training and evaluation metrics measure different aspects of performance.*

### Key Findings

- Both algorithms achieved a 100% success rate during greedy evaluation under the main configuration.
- Q-Learning achieved a slightly higher mean greedy evaluation reward and used slightly fewer steps on average.
- Q-Learning reached the selected training reward threshold earlier on average.
- Q-Learning obtained a higher average reward throughout training. The reported Welch's t-test p-value for this metric was 0.0004.
- Additional sensitivity experiments revealed that SARSA was less stable under the tested high learning-rate setting of 0.9.

These findings apply to the tested environment and experimental settings. They do not establish that Q-Learning is always better than SARSA.

## 10. Visualizations

The project generates visualizations to help interpret the training and learned behavior, including:

- Learning curves comparing rewards during training.
- Episode-step and penalty trends.
- Comparisons across five random seeds.
- Policy-arrow visualizations showing selected actions.
- Value heatmaps illustrating learned state values.
- Hyperparameter sensitivity plots.

Add your generated PNG files to the repository and embed them here. For example:

```markdown
![Learning Curves](results/learning_curves_5seeds.png)
```

Make sure the image path matches the actual location of your uploaded file.

## 11. Project Structure

A suggested repository structure is:

```text
q-learning-vs-sarsa/
├── README.md
├── q_learning_vs_sarsa.ipynb
└── results/
    ├── learning_curves_seed0.png
    ├── learning_curves_5seeds.png
    ├── policy_arrows.png
    ├── value_heatmap.png
    └── summary_5seeds.csv
```

The notebook contains the implementation, training, evaluation, and analysis. The results folder contains the generated visualizations and experiment summaries. Include only files that you have actually generated and uploaded.

## 12. How to Run the Project

The project can be run using Google Colab or a local Python environment.

### Using Google Colab

1. Open the `.ipynb` notebook in Google Colab.
2. Run the installation cell for Gymnasium if needed.
3. Execute the notebook cells in order.
4. Allow the training and evaluation experiments to finish.
5. Review the output tables, learning curves, and policy visualizations.

### Running Locally

Install the required packages:

```bash
pip install gymnasium numpy pandas matplotlib scipy
```

Then open the notebook in Jupyter Notebook or JupyterLab and run its cells in order.

## 13. Limitations

- The experiments use a discrete, tabular environment rather than a large or continuous real-world environment.
- The results depend on the chosen hyperparameters and random seeds.
- Q-tables can become impractical for environments with very large state spaces.
- Performance in Taxi-v4 does not guarantee similar results in robotics, autonomous driving, or other real-world tasks.

## 14. Future Enhancements

Possible extensions include:

- Comparing additional reinforcement learning algorithms, such as Deep Q-Networks (DQN), in suitable environments.
- Investigating a wider range of learning rates and exploration schedules.
- Testing the algorithms on more complex environments.
- Improving the visual presentation of policy and value comparisons.
- Evaluating training efficiency and stability over a larger number of random seeds.

## 15. Conclusion

This project demonstrates how Q-Learning and SARSA can learn to solve a passenger transportation task through interaction and reward feedback. Although the algorithms use different update rules, both achieved a 100% greedy evaluation success rate under the main experimental configuration.

Q-Learning achieved a higher average training reward and reached the selected reward threshold earlier in the experiments. Sensitivity analysis also highlighted differences in how the algorithms responded to a high learning rate.

Overall, the project provides a practical introduction to reinforcement learning, temporal-difference learning, policy evaluation, and experimental comparison.
