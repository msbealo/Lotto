def calculate_lotto_winnings(numbers_matched, bonus_ball_matched):
    """
    Calculate the winnings for a UK Lotto ticket based on the number of matching balls.
    
    :param numbers_matched: Number of main balls matched (0 to 6)
    :param bonus_ball_matched: Boolean indicating whether the bonus ball is matched
    :return: The amount won
    """
    if numbers_matched < 2:
        return 0
    if numbers_matched == 2:
        return 'Free Lotto Lucky Dip'  # Prize for matching 2 numbers
    if numbers_matched == 3:
        return 30  # Fixed prize for matching 3 numbers
    if numbers_matched == 4:
        return 140  # Fixed prize for matching 4 numbers
    if numbers_matched == 5:
        if bonus_ball_matched:
            return 1000000  # Prize for matching 5 numbers + bonus ball
        return 1750  # Prize for matching 5 numbers
    if numbers_matched == 6:
        return 'Jackpot'  # Varies, but indicates a jackpot win

    return 'Invalid input'

# Example usage
print(calculate_lotto_winnings(5, False))  # Match 5 numbers, no bonus ball
print(calculate_lotto_winnings(2, False))  # Match 2 numbers
print(calculate_lotto_winnings(6, False))  # Jackpot win (match 6 numbers)
