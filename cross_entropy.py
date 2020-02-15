import numpy as np

# Write a function that takes as input two lists Y, P, (where Y is the probability category and P is the probability)
# and returns the float corresponding to their cross-entropy.
def cross_entropy(Y, P):
    a = []
    CE = 0
    for i in range(len(Y)):
        b = (Y[i] * np.log(P[i])) + ((1 - Y[i]) * np.log(1 - P[i]))
        a.append(b)
    for j in range(len(a)):
        CE -= float(a[j])
    return CE
