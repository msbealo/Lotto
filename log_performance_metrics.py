def log_performance_metrics(iteration, reward, selected_numbers, num_tickets, log_interval=100):
    """
    Logs performance metrics for the model training.

    :param iteration: Current training iteration.
    :param reward: Reward obtained in the current iteration.
    :param selected_numbers: Lottery numbers selected in the current iteration.
    :param num_tickets: Number of tickets selected in the current iteration.
    :param log_interval: Interval at which to log metrics.
    """
    if iteration % log_interval == 10:
        print(f"Iteration: {iteration}, Reward: {reward}, Number of Tickets: {num_tickets}")
        print(f"Selected Numbers: {selected_numbers}")
        # Add more detailed logging here if necessary

# Example usage in the training loop
# for epoch in range(num_epochs):
    # ... [training steps] ...
#    log_performance_metrics(epoch, reward, selected_numbers, num_tickets)
