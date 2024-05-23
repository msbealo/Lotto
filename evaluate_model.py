def evaluate_model(policy_network, environment, num_trials):
    total_rewards = []
    for _ in range(num_trials):
        state = get_current_state(environment)
        selected_numbers, num_tickets = policy_network(state, training=False)
        reward = simulate_draw_and_calculate_reward(environment, selected_numbers, num_tickets)
        total_rewards.append(reward)
    
    average_reward = sum(total_rewards) / len(total_rewards)
    print(f"Average Reward after {num_trials} trials: {average_reward}")

evaluate_model(policy_network, environment, num_trials=100)
