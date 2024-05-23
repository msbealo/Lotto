import tensorflow as tf
import numpy as np

def get_current_state(environment, history_length=5):
    """
    Retrieves the current state from the LottoEnvironment.

    :param environment: Instance of LottoEnvironment.
    :param history_length: Number of past draws to consider for the state.
    :return: A numpy array representing the current state.
    """
    # Example state components
    last_winning_numbers = environment.get_last_winning_numbers(history_length)
    number_frequencies = environment.get_number_frequencies(history_length)
    last_rewards = environment.get_last_rewards(history_length)
    last_ticket_counts = environment.get_last_ticket_counts(history_length)

    # Convert these components into a numerical format suitable for the neural network
    state = []

    # Encoding last winning numbers (example: one-hot encoding)
    for numbers in last_winning_numbers:
        encoded_numbers = one_hot_encode(numbers, number_range=59)
        state.extend(encoded_numbers)
    
    # Including number frequencies, rewards, and ticket counts
    state.extend(normalize(number_frequencies))
    state.extend(normalize(last_rewards))
    state.extend(normalize(last_ticket_counts))

    state = tf.expand_dims(state, axis=0)

    return np.array(state)

def one_hot_encode(numbers, number_range):
    """One-hot encode the list of numbers."""
    encoding = [0] * number_range
    for num in numbers:
        if 1 <= num <= number_range:
            encoding[num - 1] = 1
    return encoding

def normalize(data):
    """Normalize data to a 0-1 range. Returns an empty list if data is empty."""
    if not data:
        return []
    min_val = min(data)
    max_val = max(data)
    if max_val == min_val:
        return [0.5 for _ in data]  # or another default value
    return [(x - min_val) / (max_val - min_val) for x in data]


