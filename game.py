import numpy as np
import random
from typing import Tuple, Dict, Any

class BlackjackGame:
    """
    Blackjack game environment for reinforcement learning.
    Implements a simplified version of blackjack with standard rules.
    """
    
    def __init__(self):
        self.reset()
        
    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the game to initial state.
        
        Returns:
            Tuple of (state, info)
        """
        # Deal initial cards
        self.player_cards = [self._draw_card(), self._draw_card()]
        self.dealer_cards = [self._draw_card(), self._draw_card()]
        self.deck = self._create_deck()
        
        # Calculate initial values
        self.player_sum = self._calculate_sum(self.player_cards)
        self.dealer_sum = self._calculate_sum(self.dealer_cards)
        
        # Check for natural blackjack
        if self.player_sum == 21:
            if self.dealer_sum == 21:
                self.game_over = True
                self.result = "push"
            else:
                self.game_over = True
                self.result = "win"
        elif self.dealer_sum == 21:
            self.game_over = True
            self.result = "lose"
        else:
            self.game_over = False
            self.result = None
            
        state = self._get_state()
        info = {
            "player_cards": self.player_cards.copy(),
            "dealer_cards": self.dealer_cards.copy(),
            "player_sum": self.player_sum,
            "dealer_sum": self.dealer_sum,
            "game_over": self.game_over,
            "result": self.result
        }
        
        return state, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Take an action in the game.
        
        Args:
            action: 0 for hit, 1 for stand
            
        Returns:
            Tuple of (state, reward, done, info)
        """
        if self.game_over:
            return self._get_state(), 0, True, {"result": self.result}
        
        if action == 0:  # Hit
            self.player_cards.append(self._draw_card())
            self.player_sum = self._calculate_sum(self.player_cards)
            
            if self.player_sum > 21:
                self.game_over = True
                self.result = "lose"
                reward = -1
            elif self.player_sum == 21:
                # Stand automatically
                reward = self._dealer_play()
            else:
                reward = 0
                
        elif action == 1:  # Stand
            reward = self._dealer_play()
            
        else:
            raise ValueError(f"Invalid action: {action}")
            
        state = self._get_state()
        info = {
            "player_cards": self.player_cards.copy(),
            "dealer_cards": self.dealer_cards.copy(),
            "player_sum": self.player_sum,
            "dealer_sum": self.dealer_sum,
            "game_over": self.game_over,
            "result": self.result
        }
        
        return state, reward, self.game_over, info
    
    def _dealer_play(self) -> float:
        """
        Dealer plays according to standard rules (hit on soft 17).
        Returns the reward for the player.
        """
        self.game_over = True
        
        # Dealer hits on soft 17
        while self.dealer_sum < 17 or (self.dealer_sum == 17 and self._has_ace(self.dealer_cards)):
            self.dealer_cards.append(self._draw_card())
            self.dealer_sum = self._calculate_sum(self.dealer_cards)
            
        if self.dealer_sum > 21:
            self.result = "win"
            return 1
        elif self.dealer_sum > self.player_sum:
            self.result = "lose"
            return -1
        elif self.dealer_sum < self.player_sum:
            self.result = "win"
            return 1
        else:
            self.result = "push"
            return 0
    
    def _get_state(self) -> np.ndarray:
        """
        Get the current state as a feature vector.
        
        Returns:
            State vector: [player_sum, dealer_visible_card, has_usable_ace]
        """
        dealer_visible = self.dealer_cards[0] if self.dealer_cards else 0
        has_usable_ace = self._has_usable_ace(self.player_cards)
        
        return np.array([self.player_sum, dealer_visible, has_usable_ace], dtype=np.float32)
    
    def _draw_card(self) -> int:
        """Draw a card from the deck."""
        if not self.deck:
            self.deck = self._create_deck()
        return self.deck.pop()
    
    def _create_deck(self) -> list:
        """Create a standard 52-card deck."""
        deck = []
        for suit in range(4):
            for value in range(1, 14):  # 1-13 (Ace=1, Jack=11, Queen=12, King=13)
                deck.append(value)
        random.shuffle(deck)
        return deck
    
    def _calculate_sum(self, cards: list) -> int:
        """Calculate the sum of cards, handling aces properly."""
        sum_val = 0
        num_aces = 0
        
        for card in cards:
            if card == 1:  # Ace
                num_aces += 1
                sum_val += 11
            elif card >= 10:  # Face cards
                sum_val += 10
            else:
                sum_val += card
        
        # Convert aces from 11 to 1 if needed
        while sum_val > 21 and num_aces > 0:
            sum_val -= 10
            num_aces -= 1
            
        return sum_val
    
    def _has_ace(self, cards: list) -> bool:
        """Check if hand has an ace."""
        return 1 in cards
    
    def _has_usable_ace(self, cards: list) -> bool:
        """Check if hand has a usable ace (ace that can be 11 without busting)."""
        sum_val = 0
        num_aces = 0
        
        for card in cards:
            if card == 1:
                num_aces += 1
                sum_val += 11
            elif card >= 10:
                sum_val += 10
            else:
                sum_val += card
        
        # Check if any ace can be 11
        while sum_val > 21 and num_aces > 0:
            sum_val -= 10
            num_aces -= 1
            
        return num_aces > 0
    
    def get_action_space(self) -> int:
        """Get the number of possible actions."""
        return 2  # 0: hit, 1: stand
    
    def get_state_space(self) -> int:
        """Get the dimension of the state space."""
        return 3  # player_sum, dealer_visible, has_usable_ace 