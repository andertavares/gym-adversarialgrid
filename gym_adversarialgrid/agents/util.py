import random


def categorical_draw(probabilities):
    """
    Performs a roulette-like selection (each option has a slice
    proportional to its probability)
    :param probabilities:
    :return: int - the index of the drawn choice
    """
    z = random.random()
    cum_prob = 0.0

    for choice, prob in enumerate(probabilities):
        cum_prob += prob
        if cum_prob > z:
            return choice

    print('WARNING: categorical_draw reached its end')
    return len(probabilities) - 1  # I think code should not reach here
