import random
import tensorflow as tf
import numpy as np

class LottoEnvironment:
    def __init__(self, cost_factor, jackpot_amount):
        self.cost_factor = cost_factor
        self.ticket_cost = 2  # Cost of one lotto ticket
        self.jackpot_amount = jackpot_amount
        self.prize_structure = {
            2: 2,  # Assuming Lucky Dip as £2 win
            3: 30,
            4: 140,
            5: 1750,
            5.5: 1000000,  # 5 numbers + bonus ball
            6: self.jackpot_amount
        }
        self.history_winning_numbers = []
        self.history_rewards = []
        self.history_ticket_counts = []
        # print(f"LottoEnvironment initialized with cost factor {cost_factor} and jackpot amount {jackpot_amount}")

    def generate_draw(self):
        """Generates a random lotto draw with a separate bonus ball."""
        main_draw = random.sample(range(1, 60), 6)
        remaining_numbers = [num for num in range(1, 60) if num not in main_draw]
        bonus_ball = random.choice(remaining_numbers)
        print(f"Generated Draw: {main_draw}, Bonus Ball: {bonus_ball}")
        return main_draw, bonus_ball

    def calculate_reward_old(self, ticket_numbers, winning_numbers, bonus_ball):
        """
        Calculate the reward for a given ticket based on the draw outcome.
        """
        if not self._validate_ticket(ticket_numbers):
            print("Invalid ticket.")
            return 0
    
        total_reward = 0
        for ticket in ticket_numbers:
            ticket_set = set(ticket)
            winning_set = set(winning_numbers)
            match_count = len(ticket_set & winning_set)
            reward = self.prize_structure.get(match_count, 0)
    
            if match_count == 5 and bonus_ball in ticket:
                reward = self.prize_structure[5.5]
    
            total_reward += reward
    
        net_reward = total_reward - self.ticket_cost * len(ticket_numbers) * self.cost_factor
        # print(f"Calculated reward: {net_reward} for tickets {ticket_numbers}")
        return net_reward

    def calculate_reward(self, ticket_numbers, winning_numbers, bonus_ball):
        """
        Calculate the reward for a given ticket based on the draw outcome.
        """
        print(f"Calculating reward for tickets: {ticket_numbers}, type: {type(ticket_numbers)}")
    
        if not self._validate_ticket(ticket_numbers):
            print("Invalid ticket.")
            return 0
    
        total_reward = 0
        for ticket in ticket_numbers:
            print(f"Processing ticket: {ticket}, type: {type(ticket)}")
            ticket_set = set(ticket)
            winning_set = set(winning_numbers)
            match_count = len(ticket_set & winning_set)
            reward = self.prize_structure.get(match_count, 0)
    
            if match_count == 5 and bonus_ball in ticket:
                reward = self.prize_structure[5.5]
    
            total_reward += reward
    
        net_reward = total_reward - self.ticket_cost * len(ticket_numbers) * self.cost_factor
        print(f"Calculated net reward: {net_reward} for tickets {ticket_numbers}")
        return net_reward

    def _validate_ticket(self, ticket_numbers):
        if isinstance(ticket_numbers, tf.Tensor):
            # Convert tensor to numpy array if it's not already
            ticket_numbers = ticket_numbers.numpy()
    
        # print(f"Ticket numbers before validation: {ticket_numbers}")
    
        for ticket in ticket_numbers:
            if len(ticket) != 6 or not all(1 <= num <= 59 for num in ticket) or len(set(ticket)) != 6:
                print(f"Invalid ticket detected during validation: {ticket}")
                return False
    
        # print(f"Validated ticket: {ticket_numbers}")
        return True

    def update_history(self, winning_numbers, reward, ticket_count):
        """Update historical data after each draw."""
        self.history_winning_numbers.append(winning_numbers)
        self.history_rewards.append(reward)
        self.history_ticket_counts.append(ticket_count)
        print(f"Updated history with winning numbers: {winning_numbers}, reward: {reward}, ticket count: {ticket_count}")

    def get_last_winning_numbers(self, history_length):
        """Return the last 'history_length' winning numbers."""
        last_numbers = self.history_winning_numbers[-history_length:]
        print(f"Last {history_length} winning numbers: {last_numbers}")
        return last_numbers

    def get_number_frequencies(self, history_length):
        """Calculate frequencies of each number in the last 'history_length' draws."""
        frequencies = [0] * 59  # Assuming 59 possible numbers
        for draw in self.history_winning_numbers[-history_length:]:
            for number in draw:
                frequencies[number - 1] += 1
        # print(f"Number frequencies in last {history_length} draws: {frequencies}")
        return frequencies

    def get_last_rewards(self, history_length):
        """Return the last 'history_length' rewards."""
        last_rewards = self.history_rewards[-history_length:]
        print(f"Last {history_length} rewards: {last_rewards}")
        return last_rewards

    def get_last_ticket_counts(self, history_length):
        """Return the last 'history_length' ticket counts."""
        last_counts = self.history_ticket_counts[-history_length:]
        print(f"Last {history_length} ticket counts: {last_counts}")
        return last_counts

# Example usage commented out for clarity
