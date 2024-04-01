
def pretty_params(num_params):
    """
    Pretty formats the number of parameters
    """
    if num_params < 1e6:
        return f"{num_params:.0f}"
    elif num_params < 1e9:
        return f"{num_params / 1e6:.2f}M"
    elif num_params < 1e12:
        return f"{num_params / 1e9:.2f}B"
    else:
        return f"{num_params / 1e12:.2f}T"