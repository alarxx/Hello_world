import math

def combination(n, r):
    return int((math.factorial(n)) / ((math.factorial(r)) * math.factorial(n - r)))

def pascals_triangle(window):
    cols = window-1
    row = []
    for element in range(cols + 1):
        row.append(combination(cols, element))
    return row

print(pascals_triangle(3))