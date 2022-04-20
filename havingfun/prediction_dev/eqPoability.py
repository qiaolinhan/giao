'''
spreading probability, works for states of neibour cells
P_h: original prob
P_veg, P_den: vegetation related prob (vegetation kinds, vegetation coverage dencity)
P_w: wind velosity
P_s: topography slope ralated prob
P_burn = P_h * (1 + P_veg * (1 + P_den)) * P_w * P_s
'''

import math
P_h = 0.7 # original spreading probability

def prob(P_veg, P_den, P_w, P_s):
    P_burn = P_h * (1 + P_veg * (1 + P_den)) * P_w * P_s
    return P_burn

if __name__ == "__main__":
    # params need regression or meansurement
    # P_burn works for s(x, y, t) = 1

    P_veg = 0.7 # related to different kinds of vegetation
    P_den = 0.7 # related to different vegetation cover density
    P_w = 0.7 # ralated to wind velocity
    P_s = 0.8 # related to topography slope

    P_burn = prob(P_veg, P_den, P_w, P_s)
    print("the probability of ignition is:", P_burn)