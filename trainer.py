import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any
from .game import BlackjackGame
from .agent import QLearningAgent, MonteCarloAgent

class BlackjackTrainer:
    """
    Trainer class for blackjack reinforcement learning agents.
    """
    
    def __init__(self, agent_type: str = "q_learning"):
        """
        Initialize the trainer.
        
        Args:
            agent_type: Type of agent ("q_learning" or "monte_carlo")
        """
        self.game = BlackjackGame()
        
        if agent_type == "q_learning":
            self.agent = QLearningAgent()
        elif agent_type == "monte_carlo":
            self.agent = MonteCarloAgent()
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        self.agent_type = agent_type
        self.training_stats = {
            "episode_rewards": [],
            "win_rates": [],
            "epsilon_values": []
        }
    
    def train_q_learning(self, num_episodes: int = 10000, 
                        eval_interval: int = 1000) -> Dict[str, List]:
        """
        Train the Q-learning agent.
        
        Args:
            num_episodes: Number of training episodes
            eval_interval: Interval for evaluation
            
        Returns:
            Training statistics
        """
        print(f"Training Q-Learning agent for {num_episodes} episodes...")
        
        for episode in range(num_episodes):
            state, info = self.game.reset()
            episode_reward = 0
            
            while not info["game_over"]:
                action = self.agent.get_action(state)
                next_state, reward, done, info = self.game.step(action)
                
                self.agent.update(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward
            
            # Record episode reward
            self.agent.episode_rewards.append(episode_reward)
            
            # Decay epsilon
            self.agent.decay_epsilon()
            
            # Evaluation
            if (episode + 1) % eval_interval == 0:
                win_rate = self._evaluate_agent(1000)
                self.training_stats["win_rates"].append(win_rate)
                self.training_stats["epsilon_values"].append(self.agent.epsilon)
                print(f"Episode {episode + 1}: Win Rate = {win_rate:.3f}, Epsilon = {self.agent.epsilon:.3f}")
        
        return self.training_stats
    
    def train_monte_carlo(self, num_episodes: int = 10000,
                          eval_interval: int = 1000) -> Dict[str, List]:
        """
        Train the Monte Carlo agent.
        
        Args:
            num_episodes: Number of training episodes
            eval_interval: Interval for evaluation
            
        Returns:
            Training statistics
        """
        print(f"Training Monte Carlo agent for {num_episodes} episodes...")
        
        for episode in range(num_episodes):
            state, info = self.game.reset()
            episode_reward = 0
            
            # Record initial step
            self.agent.record_step(state, 0, 0)  # Dummy action and reward for initial state
            
            while not info["game_over"]:
                action = self.agent.get_action(state)
                next_state, reward, done, info = self.game.step(action)
                
                self.agent.record_step(state, action, reward)
                state = next_state
                episode_reward += reward
            
            # Update Q-values from episode
            self.agent.update_from_episode()
            
            # Record episode reward
            self.agent.episode_rewards.append(episode_reward)
            
            # Decay epsilon
            self.agent.decay_epsilon()
            
            # Evaluation
            if (episode + 1) % eval_interval == 0:
                win_rate = self._evaluate_agent(1000)
                self.training_stats["win_rates"].append(win_rate)
                self.training_stats["epsilon_values"].append(self.agent.epsilon)
                print(f"Episode {episode + 1}: Win Rate = {win_rate:.3f}, Epsilon = {self.agent.epsilon:.3f}")
        
        return self.training_stats
    
    def train(self, num_episodes: int = 10000, eval_interval: int = 1000) -> Dict[str, List]:
        """
        Train the agent based on its type.
        
        Args:
            num_episodes: Number of training episodes
            eval_interval: Interval for evaluation
            
        Returns:
            Training statistics
        """
        if self.agent_type == "q_learning":
            return self.train_q_learning(num_episodes, eval_interval)
        else:
            return self.train_monte_carlo(num_episodes, eval_interval)
    
    def _evaluate_agent(self, num_games: int = 1000) -> float:
        """
        Evaluate the agent's performance.
        
        Args:
            num_games: Number of games to play for evaluation
            
        Returns:
            Win rate (wins / total games)
        """
        wins = 0
        
        for _ in range(num_games):
            state, info = self.game.reset()
            
            while not info["game_over"]:
                action = self.agent.get_action(state)
                state, reward, done, info = self.game.step(action)
                
                if done and reward == 1:
                    wins += 1
        
        return wins / num_games
    
    def plot_training_progress(self):
        """Plot training progress."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Win rate over time
        if self.training_stats["win_rates"]:
            episodes = range(len(self.training_stats["win_rates"])) * 1000
            ax1.plot(episodes, self.training_stats["win_rates"])
            ax1.set_xlabel("Episode")
            ax1.set_ylabel("Win Rate")
            ax1.set_title("Training Progress - Win Rate")
            ax1.grid(True)
        
        # Epsilon decay
        if self.training_stats["epsilon_values"]:
            episodes = range(len(self.training_stats["epsilon_values"])) * 1000
            ax2.plot(episodes, self.training_stats["epsilon_values"])
            ax2.set_xlabel("Episode")
            ax2.set_ylabel("Epsilon")
            ax2.set_title("Exploration Rate Decay")
            ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def save_agent(self, filename: str):
        """Save the trained agent."""
        if self.agent_type == "q_learning":
            self.agent.save_q_table(filename)
        else:
            # For Monte Carlo, save both Q-table and visit counts
            import pickle
            data = {
                "q_table": dict(self.agent.q_table),
                "visit_counts": dict(self.agent.visit_counts)
            }
            with open(filename, 'wb') as f:
                pickle.dump(data, f)
    
    def load_agent(self, filename: str):
        """Load a trained agent."""
        if self.agent_type == "q_learning":
            self.agent.load_q_table(filename)
        else:
            # For Monte Carlo, load both Q-table and visit counts
            import pickle
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.agent.q_table = data["q_table"]
                self.agent.visit_counts = data["visit_counts"]

def train_agent(agent_type: str = "q_learning", 
                num_episodes: int = 10000,
                eval_interval: int = 1000,
                save_path: str = None) -> BlackjackTrainer:
    """
    Convenience function to train an agent.
    
    Args:
        agent_type: Type of agent ("q_learning" or "monte_carlo")
        num_episodes: Number of training episodes
        eval_interval: Interval for evaluation
        save_path: Path to save the trained agent
        
    Returns:
        Trained trainer object
    """
    trainer = BlackjackTrainer(agent_type)
    stats = trainer.train(num_episodes, eval_interval)
    
    if save_path:
        trainer.save_agent(save_path)
    
    return trainer 