def count_matching_numbers(ticket_numbers, drawn_numbers):
    """
    Count the number of matching numbers between a Lotto ticket and the drawn numbers.
    
    :param ticket_numbers: A list of numbers on the Lotto ticket
    :param drawn_numbers: A list of drawn numbers
    :return: Number of matching numbers
    """
    return len(set(ticket_numbers) & set(drawn_numbers))

# Example usage
ticket_numbers = [5, 12, 23, 34, 45, 56]  # Example ticket numbers
drawn_numbers = [12, 24, 34, 45, 56, 67]  # Example drawn numbers
print(count_matching_numbers(ticket_numbers, drawn_numbers))
