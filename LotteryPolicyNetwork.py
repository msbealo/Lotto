import tensorflow as tf

class LotteryPolicyNetwork(tf.keras.Model):
    def __init__(self, state_size, number_range, max_tickets):
        super(LotteryPolicyNetwork, self).__init__()
        self.state_size = state_size
        self.number_range = number_range
        self.max_tickets = max_tickets

        # Initial layer to process the input state
        self.dense1 = tf.keras.layers.Dense(128, activation='relu', input_shape=(state_size,))

        # Additional layers for processing
        self.dense2 = tf.keras.layers.Dense(128, activation='relu')

        # Output layer for multiple ticket number sets
        # Each ticket requires a set of 6 numbers, and we prepare for max_tickets
        self.number_output = tf.keras.layers.Dense(max_tickets * 6 * number_range, activation='softmax')

    def call(self, state):
        x = self.dense1(state)
        # print(f"Dense1 layer output: {x}")
        x = self.dense2(x)
        # print(f"Dense2 layer output: {x}")
        number_logits = self.number_output(x)
        # print(f"Number output layer logits: {number_logits}")

        # Reshape to [batch_size, max_tickets, 6, number_range]
        number_probs = tf.reshape(number_logits, [-1, self.max_tickets, 6, self.number_range])
        # print(f"Reshaped number probabilities: {number_probs}")
        return number_probs

    def select_numbers(self, number_probs):
        """
        Selects unique lottery numbers based on the probability distributions for each slot.
        """
        selected_numbers = []
        for prob in number_probs:
            numbers = tf.argsort(prob)[-6:]  # Select top 6 numbers
            selected_numbers.append(numbers)
        selected_numbers_tensor = tf.stack(selected_numbers)
        # print(f"Selected numbers: {selected_numbers_tensor}")
        return selected_numbers_tensor

    def compute_log_probabilities(self, states, selected_numbers):
        """
        Compute the log probabilities of the selected lottery numbers.

        :param states: The input states to the network.
        :param selected_numbers: The selected lottery numbers for each ticket.
        :return: The log probabilities of the selected numbers.
        """
        number_probs = self.call(states)

        # Flatten the number_probs to match the shape of selected_numbers
        flat_number_probs = tf.reshape(number_probs, [-1, self.number_range])

        # Flatten selected_numbers to align with flat_number_probs for gather operation
        batch_size = tf.shape(states)[0]
        flat_indices = tf.range(0, batch_size) * self.max_tickets * 6
        flat_indices = tf.reshape(flat_indices, [-1, 1])
        flat_selected_numbers = tf.reshape(selected_numbers + flat_indices, [-1])

        # Gather the probabilities of the selected numbers
        selected_probs = tf.gather(flat_number_probs, flat_selected_numbers)

        # Compute log probabilities for selected numbers
        number_log_probs = tf.math.log(selected_probs)

        # Reshape back to the original batch shape
        number_log_probs = tf.reshape(number_log_probs, [batch_size, -1])

        # Sum the log probabilities across each ticket for each batch
        total_log_probs = tf.reduce_sum(number_log_probs, axis=1)

        return total_log_probs


# Example usage of the network
# [Code for instantiation and example usage can be included here]

