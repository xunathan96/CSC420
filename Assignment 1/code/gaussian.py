import numpy as np

def gaussian_distribution(std, x, y):
    term1 = 1/(2*np.pi*(std**2))
    term2 = np.exp(-(x**2 + y**2)/(2*(std**2)))
    return term1*term2

def gaussian_kernel(std):
    # The normal distribution is effectively zero at
    # 3 standard deviations away from the mean
    # Therefore given std we choose a kernel size of 4*std + 1

    # kernel size
    k = int(4*std + 1)  # make sure is an integer number

    # kernel sizes must be odd
    if k%2 == 0:
        k = k + 1

    # index of mean
    mu = (k-1)/2

    G = np.zeros((k,k))
    for x in range(0,k):
        for y in range(0,k):
            G[x, y] = gaussian_distribution(std, x-mu, y-mu)

    return G/G.sum()
