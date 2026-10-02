def central_difference(f, x, h):
    return (f(x + h) - f(x - h)) / (2.0 * h)

def second_difference(f, x, h):
    return (f(x + h) - 2.0 * f(x) + f(x - h)) / (h * h)
