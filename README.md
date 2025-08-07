# Blackjack Reinforcement Learning Project

A comprehensive reinforcement learning system for playing blackjack, featuring Q-learning and Monte Carlo agents with visualization and analysis tools.

## Features

- **Multiple RL Algorithms**: Q-learning and Monte Carlo agents
- **Complete Game Environment**: Full blackjack implementation with standard rules
- **Training Tools**: Easy-to-use training interface with progress tracking
- **Visualization**: Policy heatmaps, Q-value plots, and performance analysis
- **Interactive Demo**: Play against trained agents
- **Performance Analysis**: Compare with basic strategy and other agents

## Project Structure

```
Black_Jack_RL_Project/
├── blackjack/
│   ├── __init__.py          # Package initialization
│   ├── game.py              # Blackjack game environment
│   ├── agent.py             # Q-learning and Monte Carlo agents
│   ├── trainer.py           # Training utilities
│   ├── visualizer.py        # Visualization and analysis tools
│   ├── demo.py              # Interactive demo script
│   └── example.py           # Quick start examples
├── requirements.txt          # Python dependencies
└── README.md               # This file
```

## Installation

1. Clone or download the project
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from blackjack.game import BlackjackGame
from blackjack.agent import QLearningAgent
from blackjack.trainer import BlackjackTrainer

# Create game and agent
game = BlackjackGame()
agent = QLearningAgent()

# Train the agent
trainer = BlackjackTrainer("q_learning")
trainer.agent = agent
trainer.game = game
stats = trainer.train(num_episodes=5000)

# Test the agent
wins = 0
for _ in range(1000):
    state, info = game.reset()
    while not info["game_over"]:
        action = agent.get_action(state)
        state, reward, done, info = game.step(action)
        if done and reward == 1:
            wins += 1

print(f"Win rate: {wins/1000:.3f}")
```

### Run the Demo

```python
from blackjack.demo import main
main()
```

### Quick Example

```python
from blackjack.example import quick_example
agent = quick_example()
```

## Components

### Game Environment (`game.py`)

The `BlackjackGame` class implements a complete blackjack environment:

- Standard blackjack rules
- Proper ace handling (soft/hard totals)
- Dealer plays according to house rules
- Gym-like interface for RL algorithms

**State Space**: `[player_sum, dealer_visible_card, has_usable_ace]`
**Action Space**: `[0: hit, 1: stand]`
**Rewards**: `+1` for win, `-1` for loss, `0` for push

### Agents (`agent.py`)

#### QLearningAgent
- Uses Q-learning algorithm with epsilon-greedy exploration
- Configurable learning rate, discount factor, and exploration rate
- Automatic epsilon decay during training

#### MonteCarloAgent
- Uses first-visit Monte Carlo method
- Episode-based learning
- Suitable for episodic environments like blackjack

### Training (`trainer.py`)

The `BlackjackTrainer` class provides:

- Unified training interface for both agent types
- Progress tracking and evaluation
- Automatic model saving/loading
- Training statistics and visualization

### Visualization (`visualizer.py`)

The `BlackjackVisualizer` class offers:

- Policy heatmaps showing optimal actions
- Q-value visualization
- Action probability plots
- Performance comparison with basic strategy
- Win rate and expected value analysis

## Advanced Usage

### Custom Training

```python
from blackjack.trainer import train_agent

# Train Q-learning agent
q_trainer = train_agent(
    agent_type="q_learning",
    num_episodes=10000,
    eval_interval=1000,
    save_path="trained_q_agent.pkl"
)

# Train Monte Carlo agent
mc_trainer = train_agent(
    agent_type="monte_carlo",
    num_episodes=10000,
    eval_interval=1000,
    save_path="trained_mc_agent.pkl"
)
```

### Visualization and Analysis

```python
from blackjack.visualizer import BlackjackVisualizer

# Create visualizer
visualizer = BlackjackVisualizer(trained_agent)

# Plot policy
visualizer.plot_policy_heatmap()

# Plot Q-values
visualizer.plot_q_values()

# Analyze performance
performance = visualizer.analyze_optimal_play(10000)
print(f"Win rate: {performance['win_rate']:.3f}")

# Compare with basic strategy
comparison = visualizer.compare_with_basic_strategy(10000)
visualizer.plot_performance_comparison(comparison)
```

### Interactive Play

```python
from blackjack.demo import interactive_game
from blackjack.trainer import train_agent

# Train an agent
trainer = train_agent("q_learning", 2000)

# Play against it
result = interactive_game(trainer.agent)
```

## Algorithm Details

### Q-Learning
- **Update Rule**: `Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]`
- **Exploration**: Epsilon-greedy with decay
- **Advantages**: Works well with continuous training, handles exploration-exploitation trade-off

### Monte Carlo
- **Method**: First-visit Monte Carlo
- **Update Rule**: `Q(s,a) ← Q(s,a) + (G - Q(s,a)) / N(s,a)`
- **Advantages**: Unbiased estimates, good for episodic environments

## Performance

Typical performance after training:
- **Q-Learning**: ~42-45% win rate
- **Monte Carlo**: ~43-46% win rate
- **Basic Strategy**: ~42-44% win rate

Note: These are approximate values and may vary based on training parameters and random seeds.

## Customization

### Agent Parameters

```python
# Q-learning with custom parameters
agent = QLearningAgent(
    learning_rate=0.1,
    discount_factor=0.95,
    epsilon=0.1,
    epsilon_decay=0.995,
    epsilon_min=0.01
)

# Monte Carlo with custom parameters
agent = MonteCarloAgent(
    epsilon=0.1,
    epsilon_decay=0.995,
    epsilon_min=0.01
)
```

### Training Parameters

```python
trainer = BlackjackTrainer("q_learning")
stats = trainer.train(
    num_episodes=10000,
    eval_interval=1000
)
```

## Contributing

Feel free to contribute by:
- Adding new RL algorithms
- Improving visualization tools
- Optimizing performance
- Adding new features

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Inspired by classic RL problems like the one in Sutton & Barto's "Reinforcement Learning: An Introduction"
- Uses standard blackjack rules and basic strategy for comparison
- Built with modern Python libraries for scientific computing and visualization 
