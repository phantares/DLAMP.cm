def calculate_factor(number: int, k: int = 2):
    count = 0
    n = number

    while n > 1:
        n //= k
        count += 1
    scale = number / (k**count)

    return count, scale
