'''
spreading probability, works for states of neibour cells
P_h: original prob
P_veg, P_den: vegetation related prob
P_w, P_s: wind velosity, topography slope ralated prob
P_burn = P_h * (1 + P_veg * (1 + P_den)) * P_w * P_s
'''

import math
P_h = 0.7

def prob(P_veg, P_den, P_w, P_s):
    P_burn = P_h * (1 + P_veg * (1 + P_den)) * P_w * P_s
    return P_burn



if __name__ == "__main__":
    # params need regression or meansurement
    P_veg = 0.7
    P_den = 0.7
    P_w = 0.7
    P_s = 0.8
    P_burn = prob(P_veg, P_den, P_w, P_s)
    print("the probability of ignition is:", P_burn)