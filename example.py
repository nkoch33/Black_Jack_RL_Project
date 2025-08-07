#!/usr/bin/env python3
"""
Simple Blackjack RL Example

This script provides a quick example of how to use the blackjack RL system.
"""

import numpy as np
from .game import BlackjackGame
from .agent import QLearningAgent
from .trainer import BlackjackTrainer

def quick_example():
    """Quick example of training and using a blackjack agent."""
    print("=== Quick Blackjack RL Example ===")
    
    # Create game and agent
    game = BlackjackGame()
    agent = QLearningAgent(epsilon=0.1)
    
    # Train the agent
    print("Training agent for 1000 episodes...")
    trainer = BlackjackTrainer("q_learning")
    trainer.agent = agent
    trainer.game = game
    
    stats = trainer.train(num_episodes=1000, eval_interval=200)
    
    # Test the trained agent
    print("\nTesting trained agent...")
    wins = 0
    total_games = 1000
    
    for _ in range(total_games):
        state, info = game.reset()
        
        while not info["game_over"]:
            action = agent.get_action(state)
            state, reward, done, info = game.step(action)
            
            if done and reward == 1:
                wins += 1
    
    win_rate = wins / total_games
    print(f"Win rate after training: {win_rate:.3f}")
    
    # Show some example games
    print("\n=== Example Games ===")
    for i in range(3):
        print(f"\nGame {i+1}:")
        state, info = game.reset()
        print(f"Initial state: Player={info['player_sum']}, Dealer visible={info['dealer_cards'][0]}")
        
        step = 0
        while not info["game_over"]:
            step += 1
            action = agent.get_action(state)
            action_name = "HIT" if action == 0 else "STAND"
            
            state, reward, done, info = game.step(action)
            print(f"  Step {step}: {action_name} -> Player={info['player_sum']}")
            
            if done:
                result = "WIN" if reward == 1 else "LOSE" if reward == -1 else "PUSH"
                print(f"  Result: {result}")
                break
    
    return agent

def policy_example():
    """Show the learned policy for some example states."""
    print("\n=== Policy Examples ===")
    
    # Train a quick agent
    agent = quick_example()
    
    # Show policy for some example states
    example_states = [
        (12, 6, 0),  # Player 12, Dealer 6, No usable ace
        (16, 10, 0), # Player 16, Dealer 10, No usable ace
        (18, 9, 0),  # Player 18, Dealer 9, No usable ace
        (20, 5, 0),  # Player 20, Dealer 5, No usable ace
        (12, 6, 1),  # Player 12, Dealer 6, Has usable ace
        (18, 9, 1),  # Player 18, Dealer 9, Has usable ace
    ]
    
    print("Learned policy for example states:")
    print("State (Player, Dealer, Usable Ace) -> Action")
    print("-" * 50)
    
    for player_sum, dealer_card, has_usable_ace in example_states:
        state = np.array([player_sum, dealer_card, has_usable_ace], dtype=np.float32)
        action = agent.get_action(state)
        action_name = "HIT" if action == 0 else "STAND"
        
        ace_text = "Yes" if has_usable_ace else "No"
        print(f"({player_sum:2d}, {dealer_card:2d}, {ace_text:3s}) -> {action_name}")
    
    return agent

if __name__ == "__main__":
    # Run the quick example
    agent = quick_example()
    
    # Show policy examples
    policy_example() 