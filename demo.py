#!/usr/bin/env python3
"""
Blackjack Reinforcement Learning Demo

This script demonstrates how to use the blackjack RL system.
It includes training examples, visualization, and interactive gameplay.
"""

import numpy as np
import matplotlib.pyplot as plt
from .game import BlackjackGame
from .agent import QLearningAgent, MonteCarloAgent
from .trainer import BlackjackTrainer, train_agent
from .visualizer import BlackjackVisualizer

def demo_training():
    """Demonstrate training a Q-learning agent."""
    print("=== Blackjack RL Training Demo ===")
    print("Training a Q-learning agent...")
    
    # Train the agent
    trainer = train_agent(
        agent_type="q_learning",
        num_episodes=5000,
        eval_interval=500,
        save_path="trained_q_agent.pkl"
    )
    
    # Plot training progress
    trainer.plot_training_progress()
    
    return trainer

def demo_monte_carlo():
    """Demonstrate training a Monte Carlo agent."""
    print("=== Monte Carlo Training Demo ===")
    print("Training a Monte Carlo agent...")
    
    # Train the agent
    trainer = train_agent(
        agent_type="monte_carlo",
        num_episodes=5000,
        eval_interval=500,
        save_path="trained_mc_agent.pkl"
    )
    
    # Plot training progress
    trainer.plot_training_progress()
    
    return trainer

def demo_visualization(trainer):
    """Demonstrate visualization of the trained agent."""
    print("=== Visualization Demo ===")
    
    # Create visualizer
    visualizer = BlackjackVisualizer(trainer.agent)
    
    # Plot policy heatmap
    print("Plotting policy heatmap...")
    visualizer.plot_policy_heatmap()
    
    # Plot Q-values
    print("Plotting Q-values...")
    visualizer.plot_q_values()
    
    # Plot action probabilities
    print("Plotting action probabilities...")
    visualizer.plot_action_probabilities()
    
    # Analyze performance
    print("Analyzing performance...")
    performance = visualizer.analyze_optimal_play(10000)
    print(f"Win rate: {performance['win_rate']:.3f}")
    print(f"Expected value: {performance['expected_value']:.3f}")
    
    # Compare with basic strategy
    print("Comparing with basic strategy...")
    comparison = visualizer.compare_with_basic_strategy(10000)
    print(f"Trained agent win rate: {comparison['trained_agent']['win_rate']:.3f}")
    print(f"Basic strategy win rate: {comparison['basic_strategy']['win_rate']:.3f}")
    
    # Plot comparison
    visualizer.plot_performance_comparison(comparison)

def interactive_game(agent):
    """Play an interactive game against the trained agent."""
    print("=== Interactive Blackjack Game ===")
    print("You'll play as the dealer, the AI will play as the player.")
    print("Press Enter to continue each step...")
    
    game = BlackjackGame()
    state, info = game.reset()
    
    print(f"\nInitial state:")
    print(f"Player cards: {info['player_cards']} (sum: {info['player_sum']})")
    print(f"Dealer cards: {info['dealer_cards']} (sum: {info['dealer_sum']})")
    print(f"Dealer visible card: {info['dealer_cards'][0]}")
    
    step = 0
    while not info["game_over"]:
        step += 1
        print(f"\n--- Step {step} ---")
        
        # AI makes decision
        action = agent.get_action(state)
        action_name = "HIT" if action == 0 else "STAND"
        print(f"AI chooses to: {action_name}")
        
        # Take action
        state, reward, done, info = game.step(action)
        
        print(f"Player cards: {info['player_cards']} (sum: {info['player_sum']})")
        print(f"Dealer cards: {info['dealer_cards']} (sum: {info['dealer_sum']})")
        
        if done:
            print(f"\nGame over! Result: {info['result'].upper()}")
            if reward == 1:
                print("AI wins!")
            elif reward == -1:
                print("Dealer wins!")
            else:
                print("Push (tie)!")
            break
        
        input("Press Enter to continue...")
    
    return info["result"]

def demo_comparison():
    """Compare Q-learning and Monte Carlo agents."""
    print("=== Agent Comparison Demo ===")
    
    # Train both agents
    print("Training Q-learning agent...")
    q_trainer = train_agent("q_learning", 3000, 500)
    
    print("Training Monte Carlo agent...")
    mc_trainer = train_agent("monte_carlo", 3000, 500)
    
    # Compare performance
    q_visualizer = BlackjackVisualizer(q_trainer.agent)
    mc_visualizer = BlackjackVisualizer(mc_trainer.agent)
    
    q_performance = q_visualizer.analyze_optimal_play(5000)
    mc_performance = mc_visualizer.analyze_optimal_play(5000)
    
    print(f"\nQ-Learning Performance:")
    print(f"  Win rate: {q_performance['win_rate']:.3f}")
    print(f"  Expected value: {q_performance['expected_value']:.3f}")
    
    print(f"\nMonte Carlo Performance:")
    print(f"  Win rate: {mc_performance['win_rate']:.3f}")
    print(f"  Expected value: {mc_performance['expected_value']:.3f}")
    
    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Win rates
    agents = ['Q-Learning', 'Monte Carlo']
    win_rates = [q_performance['win_rate'], mc_performance['win_rate']]
    expected_values = [q_performance['expected_value'], mc_performance['expected_value']]
    
    bars1 = ax1.bar(agents, win_rates, color=['blue', 'green'])
    ax1.set_ylabel('Win Rate')
    ax1.set_title('Win Rate Comparison')
    ax1.set_ylim(0, 1)
    
    for bar, rate in zip(bars1, win_rates):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{rate:.3f}', ha='center', va='bottom')
    
    # Expected values
    bars2 = ax2.bar(agents, expected_values, color=['blue', 'green'])
    ax2.set_ylabel('Expected Value')
    ax2.set_title('Expected Value Comparison')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    for bar, value in zip(bars2, expected_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.01 if value >= 0 else -0.01),
                f'{value:.3f}', ha='center', va='bottom' if value >= 0 else 'top')
    
    plt.tight_layout()
    plt.show()

def main():
    """Main demo function."""
    print("Welcome to the Blackjack Reinforcement Learning Demo!")
    print("This demo will show you how to train and use RL agents to play blackjack.")
    
    while True:
        print("\n" + "="*50)
        print("Choose a demo:")
        print("1. Train Q-learning agent")
        print("2. Train Monte Carlo agent")
        print("3. Compare both agents")
        print("4. Interactive game")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            trainer = demo_training()
            demo_visualization(trainer)
            
        elif choice == "2":
            trainer = demo_monte_carlo()
            demo_visualization(trainer)
            
        elif choice == "3":
            demo_comparison()
            
        elif choice == "4":
            # Train a quick agent for interactive play
            print("Training a quick agent for interactive play...")
            trainer = train_agent("q_learning", 2000, 500)
            interactive_game(trainer.agent)
            
        elif choice == "5":
            print("Goodbye!")
            break
            
        else:
            print("Invalid choice. Please enter 1-5.")

if __name__ == "__main__":
    main() 