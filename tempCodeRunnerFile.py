
def normalize(value, min_value=0, max_value=300):
    # Ensure value is within the given range
    value = max(min_value, min(value, max_value))
    