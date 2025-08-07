import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from .game import BlackjackGame
from .agent import QLearningAgent, MonteCarloAgent

class BlackjackVisualizer:
    """
    Visualization tools for blackjack reinforcement learning results.
    """
    
    def __init__(self, agent, game: BlackjackGame = None):
        """
        Initialize the visualizer.
        
        Args:
            agent: Trained agent (QLearningAgent or MonteCarloAgent)
            game: BlackjackGame instance (optional)
        """
        self.agent = agent
        self.game = game if game else BlackjackGame()
        
    def plot_policy_heatmap(self, figsize: Tuple[int, int] = (15, 10)):
        """
        Plot the agent's policy as a heatmap.
        
        Args:
            figsize: Figure size
        """
        # Create state space
        player_sums = range(4, 22)  # 4-21
        dealer_cards = range(1, 11)  # 1-10 (Ace=1, 10=10)
        usable_ace_options = [0, 1]  # No usable ace, Has usable ace
        
        # Create subplots
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        for ace_idx, has_usable_ace in enumerate(usable_ace_options):
            policy_matrix = np.zeros((len(player_sums), len(dealer_cards)))
            
            for i, player_sum in enumerate(player_sums):
                for j, dealer_card in enumerate(dealer_cards):
                    state = np.array([player_sum, dealer_card, has_usable_ace], dtype=np.float32)
                    action = self.agent.get_action(state)
                    policy_matrix[i, j] = action  # 0 for hit, 1 for stand
            
            # Plot heatmap
            sns.heatmap(policy_matrix, 
                       xticklabels=dealer_cards,
                       yticklabels=player_sums,
                       ax=axes[ace_idx],
                       cmap='RdYlBu_r',
                       cbar_kws={'label': 'Action (0=Hit, 1=Stand)'})
            
            ace_text = "with" if has_usable_ace else "without"
            axes[ace_idx].set_title(f'Policy {ace_text} usable ace')
            axes[ace_idx].set_xlabel('Dealer visible card')
            axes[ace_idx].set_ylabel('Player sum')
        
        plt.tight_layout()
        plt.show()
    
    def plot_q_values(self, figsize: Tuple[int, int] = (15, 10)):
        """
        Plot Q-values as heatmaps.
        
        Args:
            figsize: Figure size
        """
        # Create state space
        player_sums = range(4, 22)
        dealer_cards = range(1, 11)
        usable_ace_options = [0, 1]
        
        # Create subplots for hit and stand actions
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        for ace_idx, has_usable_ace in enumerate(usable_ace_options):
            hit_matrix = np.zeros((len(player_sums), len(dealer_cards)))
            stand_matrix = np.zeros((len(player_sums), len(dealer_cards)))
            
            for i, player_sum in enumerate(player_sums):
                for j, dealer_card in enumerate(dealer_cards):
                    state = np.array([player_sum, dealer_card, has_usable_ace], dtype=np.float32)
                    state_key = self.agent._state_to_key(state)
                    
                    hit_matrix[i, j] = self.agent.q_table[state_key][0]  # Hit Q-value
                    stand_matrix[i, j] = self.agent.q_table[state_key][1]  # Stand Q-value
            
            # Plot hit Q-values
            sns.heatmap(hit_matrix,
                       xticklabels=dealer_cards,
                       yticklabels=player_sums,
                       ax=axes[ace_idx, 0],
                       cmap='viridis',
                       cbar_kws={'label': 'Q-value'})
            ace_text = "with" if has_usable_ace else "without"
            axes[ace_idx, 0].set_title(f'Hit Q-values {ace_text} usable ace')
            axes[ace_idx, 0].set_xlabel('Dealer visible card')
            axes[ace_idx, 0].set_ylabel('Player sum')
            
            # Plot stand Q-values
            sns.heatmap(stand_matrix,
                       xticklabels=dealer_cards,
                       yticklabels=player_sums,
                       ax=axes[ace_idx, 1],
                       cmap='viridis',
                       cbar_kws={'label': 'Q-value'})
            axes[ace_idx, 1].set_title(f'Stand Q-values {ace_text} usable ace')
            axes[ace_idx, 1].set_xlabel('Dealer visible card')
            axes[ace_idx, 1].set_ylabel('Player sum')
        
        plt.tight_layout()
        plt.show()
    
    def plot_action_probabilities(self, figsize: Tuple[int, int] = (15, 10)):
        """
        Plot action probabilities as heatmaps.
        
        Args:
            figsize: Figure size
        """
        # Create state space
        player_sums = range(4, 22)
        dealer_cards = range(1, 11)
        usable_ace_options = [0, 1]
        
        # Create subplots for hit and stand probabilities
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        for ace_idx, has_usable_ace in enumerate(usable_ace_options):
            hit_probs = np.zeros((len(player_sums), len(dealer_cards)))
            stand_probs = np.zeros((len(player_sums), len(dealer_cards)))
            
            for i, player_sum in enumerate(player_sums):
                for j, dealer_card in enumerate(dealer_cards):
                    state = np.array([player_sum, dealer_card, has_usable_ace], dtype=np.float32)
                    probs = self.agent.get_policy(state)
                    
                    hit_probs[i, j] = probs[0]  # Hit probability
                    stand_probs[i, j] = probs[1]  # Stand probability
            
            # Plot hit probabilities
            sns.heatmap(hit_probs,
                       xticklabels=dealer_cards,
                       yticklabels=player_sums,
                       ax=axes[ace_idx, 0],
                       cmap='Blues',
                       cbar_kws={'label': 'Probability'})
            ace_text = "with" if has_usable_ace else "without"
            axes[ace_idx, 0].set_title(f'Hit probabilities {ace_text} usable ace')
            axes[ace_idx, 0].set_xlabel('Dealer visible card')
            axes[ace_idx, 0].set_ylabel('Player sum')
            
            # Plot stand probabilities
            sns.heatmap(stand_probs,
                       xticklabels=dealer_cards,
                       yticklabels=player_sums,
                       ax=axes[ace_idx, 1],
                       cmap='Reds',
                       cbar_kws={'label': 'Probability'})
            axes[ace_idx, 1].set_title(f'Stand probabilities {ace_text} usable ace')
            axes[ace_idx, 1].set_xlabel('Dealer visible card')
            axes[ace_idx, 1].set_ylabel('Player sum')
        
        plt.tight_layout()
        plt.show()
    
    def analyze_optimal_play(self, num_games: int = 10000) -> Dict[str, float]:
        """
        Analyze the agent's performance and compare with basic strategy.
        
        Args:
            num_games: Number of games to simulate
            
        Returns:
            Dictionary with performance metrics
        """
        wins = 0
        total_games = 0
        
        for _ in range(num_games):
            state, info = self.game.reset()
            
            while not info["game_over"]:
                action = self.agent.get_action(state)
                state, reward, done, info = self.game.step(action)
                
                if done:
                    total_games += 1
                    if reward == 1:
                        wins += 1
        
        win_rate = wins / total_games if total_games > 0 else 0
        expected_value = (wins - (total_games - wins)) / total_games if total_games > 0 else 0
        
        return {
            "win_rate": win_rate,
            "expected_value": expected_value,
            "total_games": total_games,
            "wins": wins
        }
    
    def compare_with_basic_strategy(self, num_games: int = 10000) -> Dict[str, Dict[str, float]]:
        """
        Compare the trained agent with basic blackjack strategy.
        
        Args:
            num_games: Number of games to simulate
            
        Returns:
            Dictionary with comparison results
        """
        # Basic strategy rules (simplified)
        def basic_strategy_action(player_sum: int, dealer_card: int, has_usable_ace: bool) -> int:
            if has_usable_ace:
                # Soft totals
                if player_sum >= 19:
                    return 1  # Stand
                elif player_sum == 18 and dealer_card >= 9:
                    return 0  # Hit
                elif player_sum == 18:
                    return 1  # Stand
                else:
                    return 0  # Hit
            else:
                # Hard totals
                if player_sum >= 17:
                    return 1  # Stand
                elif player_sum == 16 and dealer_card <= 6:
                    return 1  # Stand
                elif player_sum == 16:
                    return 0  # Hit
                elif player_sum == 15 and dealer_card <= 6:
                    return 1  # Stand
                elif player_sum == 15:
                    return 0  # Hit
                elif player_sum == 13 and dealer_card <= 6:
                    return 1  # Stand
                elif player_sum == 13:
                    return 0  # Hit
                elif player_sum == 12 and dealer_card <= 3:
                    return 0  # Hit
                elif player_sum == 12 and dealer_card <= 6:
                    return 1  # Stand
                elif player_sum == 12:
                    return 0  # Hit
                else:
                    return 0  # Hit
        
        # Test trained agent
        trained_wins = 0
        trained_games = 0
        
        for _ in range(num_games):
            state, info = self.game.reset()
            
            while not info["game_over"]:
                action = self.agent.get_action(state)
                state, reward, done, info = self.game.step(action)
                
                if done:
                    trained_games += 1
                    if reward == 1:
                        trained_wins += 1
        
        # Test basic strategy
        basic_wins = 0
        basic_games = 0
        
        for _ in range(num_games):
            state, info = self.game.reset()
            
            while not info["game_over"]:
                player_sum = int(state[0])
                dealer_card = int(state[1])
                has_usable_ace = bool(state[2])
                
                action = basic_strategy_action(player_sum, dealer_card, has_usable_ace)
                state, reward, done, info = self.game.step(action)
                
                if done:
                    basic_games += 1
                    if reward == 1:
                        basic_wins += 1
        
        return {
            "trained_agent": {
                "win_rate": trained_wins / trained_games if trained_games > 0 else 0,
                "expected_value": (trained_wins - (trained_games - trained_wins)) / trained_games if trained_games > 0 else 0,
                "total_games": trained_games,
                "wins": trained_wins
            },
            "basic_strategy": {
                "win_rate": basic_wins / basic_games if basic_games > 0 else 0,
                "expected_value": (basic_wins - (basic_games - basic_wins)) / basic_games if basic_games > 0 else 0,
                "total_games": basic_games,
                "wins": basic_wins
            }
        }
    
    def plot_performance_comparison(self, comparison_results: Dict[str, Dict[str, float]]):
        """
        Plot performance comparison between trained agent and basic strategy.
        
        Args:
            comparison_results: Results from compare_with_basic_strategy
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Win rates
        agents = list(comparison_results.keys())
        win_rates = [comparison_results[agent]["win_rate"] for agent in agents]
        
        bars1 = ax1.bar(agents, win_rates, color=['blue', 'orange'])
        ax1.set_ylabel('Win Rate')
        ax1.set_title('Win Rate Comparison')
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, rate in zip(bars1, win_rates):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{rate:.3f}', ha='center', va='bottom')
        
        # Expected values
        expected_values = [comparison_results[agent]["expected_value"] for agent in agents]
        
        bars2 = ax2.bar(agents, expected_values, color=['blue', 'orange'])
        ax2.set_ylabel('Expected Value')
        ax2.set_title('Expected Value Comparison')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars2, expected_values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.01 if value >= 0 else -0.01),
                    f'{value:.3f}', ha='center', va='bottom' if value >= 0 else 'top')
        
        plt.tight_layout()
        plt.show() 