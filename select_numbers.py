import tensorflow as tf
import numpy as np

def select_numbers(number_probs, number_range=59):
    """
    Selects and sorts the 6 highest probability unique numbers for each lottery ticket.

    :param number_probs: Probability distribution for each number, shape [batch_size, max_tickets, 6, number_range]
    :param number_range: Total range of numbers (e.g., 59 for a typical lottery).
    :return: Selected and sorted numbers for each ticket, shape [batch_size, max_tickets, 6]
    """
    selected_numbers = []
    for batch in tf.unstack(number_probs, axis=0):  # Iterate over the batch
        batch_numbers = []
        for ticket_probs in tf.unstack(batch, axis=0):  # Iterate over tickets
            flat_probs = tf.reshape(ticket_probs, [-1])
            ticket_numbers = set()

            while len(ticket_numbers) < 6:
                idx = tf.argmax(flat_probs).numpy()
                number = idx % number_range + 1

                # Set the probability of this number to -1 to ensure it's not picked again
                mask = tf.one_hot(idx, depth=flat_probs.shape[0], on_value=-1.0, off_value=0.0)
                flat_probs += mask

                ticket_numbers.add(number)

            batch_numbers.append(sorted(ticket_numbers))  # Sort numbers in ascending order
        selected_numbers.append(batch_numbers)

    return np.array(selected_numbers)

