import tensorflow as tf

def calculate_loss(policy_network, states, selected_numbers, rewards):
    """
    Calculates the loss for a reinforcement learning model.

    :param policy_network: The neural network representing the policy.
    :param states: A batch of states observed.
    :param actions: The actions taken by the policy network for those states (selected lottery numbers).
    :param selected_ticket_counts: The number of tickets selected for each batch.
    :param rewards: The rewards received for taking those actions.
    :return: The computed loss value.
    """
   
    log_probs = policy_network.compute_log_probabilities(states, selected_numbers)
    loss_values = -log_probs * rewards  # Negative sign for gradient ascent
    return tf.reduce_mean(loss_values)

