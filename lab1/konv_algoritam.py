#https://www.analyticsvidhya.com/blog/2022/05/knapsack-problem-in-python/
import random
import time

def knapSack(W, wt, val, n):

    K = [[0 for x in range(W + 1)] for x in range(n + 1)]

    for i in range(n + 1):
         for w in range(W + 1):
             if i == 0  or  w == 0:
                 K[i][w] = 0
             elif wt[i-1] <= w:
                 K[i][w] = max(val[i-1]
                           + K[i-1][w-wt[i-1]],
                               K[i-1][w])
             else:
                 K[i][w] = K[i-1][w]
    return K[n][W]


n = 10000
val = [random.randint(1, 100) for x in range(n + 1)]
wt = [random.randint(1, 10) for x in range(n + 1)]
W = 50000


start_time = time.time()

print(knapSack(W, wt, val, n))

print("--- %s seconds ---" % (time.time() - start_time))