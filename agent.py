import numpy as np
import random
from typing import Dict, Tuple, List
from collections import defaultdict

class QLearningAgent:
    """
    Q-Learning agent for playing blackjack.
    Uses epsilon-greedy exploration strategy.
    """
    
    def __init__(self, 
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.95,
                 epsilon: float = 0.1,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01):
        """
        Initialize the Q-Learning agent.
        
        Args:
            learning_rate: Learning rate for Q-value updates
            discount_factor: Discount factor for future rewards
            epsilon: Initial exploration rate
            epsilon_decay: Rate at which epsilon decays
            epsilon_min: Minimum epsilon value
        """
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-table: state -> action -> Q-value
        self.q_table = defaultdict(lambda: defaultdict(float))
        
        # Training statistics
        self.training_rewards = []
        self.episode_rewards = []
        
    def get_action(self, state: np.ndarray) -> int:
        """
        Choose an action using epsilon-greedy policy.
        
        Args:
            state: Current game state
            
        Returns:
            Action to take (0: hit, 1: stand)
        """
        state_key = self._state_to_key(state)
        
        # Epsilon-greedy exploration
        if random.random() < self.epsilon:
            return random.randint(0, 1)
        
        # Exploit learned Q-values
        q_values = [self.q_table[state_key][action] for action in range(2)]
        return np.argmax(q_values)
    
    def update(self, state: np.ndarray, action: int, reward: float, 
               next_state: np.ndarray, done: bool):
        """
        Update Q-values using Q-learning algorithm.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        state_key = self._state_to_key(state)
        next_state_key = self._state_to_key(next_state)
        
        # Current Q-value
        current_q = self.q_table[state_key][action]
        
        # Maximum Q-value for next state
        if done:
            max_next_q = 0
        else:
            max_next_q = max([self.q_table[next_state_key][a] for a in range(2)])
        
        # Q-learning update rule
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[state_key][action] = new_q
    
    def _state_to_key(self, state: np.ndarray) -> str:
        """
        Convert state array to string key for Q-table.
        
        Args:
            state: State array [player_sum, dealer_visible, has_usable_ace]
            
        Returns:
            String representation of state
        """
        return f"{int(state[0])},{int(state[1])},{int(state[2])}"
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def get_policy(self, state: np.ndarray) -> List[float]:
        """
        Get action probabilities for a state.
        
        Args:
            state: Current state
            
        Returns:
            List of action probabilities [hit_prob, stand_prob]
        """
        state_key = self._state_to_key(state)
        q_values = [self.q_table[state_key][action] for action in range(2)]
        
        # Convert to probabilities using softmax
        exp_q = np.exp(q_values - np.max(q_values))  # Subtract max for numerical stability
        probs = exp_q / np.sum(exp_q)
        
        return probs.tolist()
    
    def save_q_table(self, filename: str):
        """Save Q-table to file."""
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump(dict(self.q_table), f)
    
    def load_q_table(self, filename: str):
        """Load Q-table from file."""
        import pickle
        with open(filename, 'rb') as f:
            q_table_dict = pickle.load(f)
            self.q_table = defaultdict(lambda: defaultdict(float), q_table_dict)

class MonteCarloAgent:
    """
    Monte Carlo agent for playing blackjack.
    Uses first-visit Monte Carlo method.
    """
    
    def __init__(self, epsilon: float = 0.1, epsilon_decay: float = 0.995, epsilon_min: float = 0.01):
        """
        Initialize the Monte Carlo agent.
        
        Args:
            epsilon: Initial exploration rate
            epsilon_decay: Rate at which epsilon decays
            epsilon_min: Minimum epsilon value
        """
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-table and visit counts
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.visit_counts = defaultdict(lambda: defaultdict(int))
        
        # Episode buffer
        self.episode_buffer = []
        
    def get_action(self, state: np.ndarray) -> int:
        """
        Choose an action using epsilon-greedy policy.
        
        Args:
            state: Current game state
            
        Returns:
            Action to take (0: hit, 1: stand)
        """
        state_key = self._state_to_key(state)
        
        # Epsilon-greedy exploration
        if random.random() < self.epsilon:
            return random.randint(0, 1)
        
        # Exploit learned Q-values
        q_values = [self.q_table[state_key][action] for action in range(2)]
        return np.argmax(q_values)
    
    def record_step(self, state: np.ndarray, action: int, reward: float):
        """
        Record a step in the current episode.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
        """
        self.episode_buffer.append((state, action, reward))
    
    def update_from_episode(self):
        """
        Update Q-values from the recorded episode using Monte Carlo method.
        """
        if not self.episode_buffer:
            return
        
        # Calculate returns for each step
        returns = []
        total_return = 0
        
        for _, _, reward in reversed(self.episode_buffer):
            total_return += reward
            returns.insert(0, total_return)
        
        # Update Q-values
        visited_states = set()
        
        for i, (state, action, _) in enumerate(self.episode_buffer):
            state_key = self._state_to_key(state)
            state_action = (state_key, action)
            
            # First-visit Monte Carlo: only update first occurrence
            if state_action not in visited_states:
                visited_states.add(state_action)
                
                # Incremental update
                self.visit_counts[state_key][action] += 1
                count = self.visit_counts[state_key][action]
                
                current_q = self.q_table[state_key][action]
                new_q = current_q + (returns[i] - current_q) / count
                self.q_table[state_key][action] = new_q
        
        # Clear episode buffer
        self.episode_buffer = []
    
    def _state_to_key(self, state: np.ndarray) -> str:
        """
        Convert state array to string key for Q-table.
        
        Args:
            state: State array [player_sum, dealer_visible, has_usable_ace]
            
        Returns:
            String representation of state
        """
        return f"{int(state[0])},{int(state[1])},{int(state[2])}"
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay) 