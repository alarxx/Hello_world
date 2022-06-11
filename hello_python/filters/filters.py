import numpy as np
import math

def expand_border(x, border_size, left_v=None, right_v=None):
    if not left_v:
        left_v = x[0]
    if not right_v:
        right_v = x[len(x) - 1]

    left_v = np.full(border_size, left_v)
    right_v = np.full(border_size, right_v)

    x = np.insert(x, 0, left_v)
    x = np.append(x, right_v)

    return x

def median_filter(x, window):
    border_size = window // 2
    x = expand_border(x, border_size=border_size)
    result = np.array([])
    for i in range(len(x)-border_size*2):
        a = x[i : i+window]
        a = np.sort(a)
        result = np.append(result, a[border_size])
    return result

def max_median_filter(x, window):
    border_size = window // 2
    x = expand_border(x, border_size=border_size)
    result = np.array([])
    for i in range(len(x)-border_size*2):
        a = x[i : i+window]
        val = a[border_size]
        a = np.sort(a)
        if val != a[window-1]:
            val = a[border_size]
        result = np.append(result, val)
    return result


def combination(n, r):
    return int((math.factorial(n)) / ((math.factorial(r)) * math.factorial(n - r)))

def pascals_triangle(window):
    cols = window-1
    row = []
    for element in range(cols + 1):
        row.append(combination(cols, element))
    return row

def gaussian_filter(x, window):
    kernel = np.asarray(pascals_triangle(window))
    x = np.convolve(x, kernel, mode='same')
    x = np.divide(x, np.sum(kernel))

    return x


def average_filter(x, window):
    border_size = window // 2
    x = expand_border(x, border_size=border_size)
    result = np.array([])
    for i in range(len(x) - border_size * 2):
        a = x[i: i + window]
        result = np.append(result, np.sum(a)/window)
    return result


def gradient_filter(x, window):
    border_size = window // 2
    x = expand_border(x, border_size=border_size)
    result = np.array([])
    for i in range(len(x) - border_size * 2):
        a = x[i: i + window]

        minus = a[:border_size]
        plus = a[border_size+1:]

        val = np.sum(plus) - np.sum(minus)

        result = np.append(result, val)
    return result


def compressor(x, step):
    result = np.array([])

    for i in range(len(x)//step):

        a = x[i*step: i*step + step]
        result = np.append(result, np.sum(a) / step)

    if len(x) % step != 0:
        a = x[len(x) - len(x) % step:]
        result = np.append(result, np.sum(a) / len(a))

    return result


def median_compressor(x, step):
    result = np.array([])

    for i in range(len(x)//step):
        a = x[i*step: i*step + step]

        a = np.sort(a)
        result = np.append(result, a[step // 2])

    if len(x) % step != 0:
        a = x[len(x) - len(x) % step:]

        a = np.sort(a)
        result = np.append(result, a[len(a) // 2])

    return result

def step_compressor(x, window, step):
    result = np.array([])
    for i in range(((len(x)-window) // step) + 1):
        a = x[i * step: i * step + window]
        result = np.append(result, np.sum(a) / window)

    if len(x)-window < 0:
        a = x[:]
        result = np.append(result, np.sum(a) / len(a))
    elif (len(x)-window) % step != 0:
        a = x[len(x) - (len(x)-window) % step:]
        result = np.append(result, np.sum(a) / len(a))

    return result


def step_median_compressor(x, window, step):
    result = np.array([])
    for i in range(((len(x) - window) // step) + 1):
        a = x[i * step: i * step + window]
        a = np.sort(a)
        result = np.append(result, a[len(a) // 2])

    if len(x) - window < 0:
        a = x[:]
        a = np.sort(a)
        result = np.append(result, a[len(a) // 2])
    elif (len(x) - window) % step != 0:
        a = x[len(x) - (len(x) - window) % step:]

        a = np.sort(a)
        result = np.append(result, a[len(a)//2])

    return result


if __name__ == "__main__":
    window = 5

    x = np.array([1, 2, 3, 4, 5, 6, 7])

    x = step_compressor(x, 5, 2)

    print(x)