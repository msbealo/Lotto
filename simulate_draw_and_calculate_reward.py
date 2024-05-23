import numpy as np

def simulate_draw_and_calculate_reward(environment, selected_numbers):
    """
    Simulates a lottery draw and calculates the reward for each ticket.

    :param environment: An instance of LottoEnvironment.
    :param selected_numbers: An array of selected lottery numbers, shape [batch_size, max_tickets, 6].
    :return: The total reward from all tickets.
    """
    # Initialize total reward
    total_reward = 0

    # Simulate the lottery draw
    winning_numbers, bonus_ball = environment.generate_draw()
    print(f"Generated Draw: {winning_numbers}, Bonus Ball: {bonus_ball}")

    # Print the structure of selected_numbers for debugging
    print(f"Selected Numbers Structure: {type(selected_numbers)} - {np.shape(selected_numbers)}")

    # Iterate over each ticket's selected numbers
    for ticket_numbers in selected_numbers[0]:  # Assuming selected_numbers is [batch_size, max_tickets, 6]
        print(f"Processing Ticket Numbers: {ticket_numbers}")

        # Ensure ticket_numbers is formatted as a list of integers
        if isinstance(ticket_numbers, np.ndarray):
            ticket_numbers_list = ticket_numbers.tolist()
        elif isinstance(ticket_numbers, list):
            ticket_numbers_list = ticket_numbers
        else:
            raise TypeError("ticket_numbers must be a list or numpy array")

        # Calculate the reward for this ticket
        reward = environment.calculate_reward(ticket_numbers_list, winning_numbers, bonus_ball)
        print(f"Ticket Numbers: {ticket_numbers_list}, Reward: {reward}")

        # Update total reward
        total_reward += reward

    # Update environment's history (optional)
    environment.update_history(winning_numbers, total_reward, len(selected_numbers[0]))

    return total_reward
