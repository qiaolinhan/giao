'''
rate of spreading, works inside of each cell, 
to calculate how long it will take for state changing
probability of a cell can being fuel, or it is clear
P_burn = P_origial * (1+P_veg(1+P_density)) * P_wind * P_slope
if |theta| < pi/2: U(V, theta) = epsilon_0 + alpha * \sqrt(V*cos^v(theta))
if |theta| > pi>2: U(V, theta) = epsilon_0 * (beta + (1-beta)*|sin(theta)|)
if slope: U(V, theta) = U * e^{2s}
'''

import math

# params need regression
epsilon_0 = 11
alpha = 1
beta = 1
n = 2 # times

# rate of spreading
def ros(V=20, theta=45, s=None):
    if abs(theta) <= math.pi/2:
        u = epsilon_0 + alpha * math.sqrt(V * math.pow(math.cos(theta), n))
    if abs(theta) > math.pi/2:
        u = epsilon_0 * (beta + (1 - beta) * abs(math.sin(theta)))
    u_mid = u
    if s is not None:
        u_slope = u * math.exp(2 * s)
        return u_slope
    else:
        u_flat = u_mid
        return u_flat

def n_step(u_test=100, total_time = 200, cell_size = 100):
    each_time = cell_size/u_test
    n_step = round(total_time/each_time)
    return n_step

# spreading probability
# P_h: original prob
# P_veg, P_den: vegetation related prob
# P_w, P_s: wind velosity, topography slope ralated prob
# P_burn = P_h * (1 + P_veg * (1 + P_den)) * P_w * P_s
if __name__ == "__main__":
    V = 200 # m/s
    theta = math.radians(45) # between wind direction and normal spreading direction
    s = math.radians(15) # slope of topography
    u_test= ros(V, theta, s)
    print("The rate of spreading is:", u_test)

    total_time = 300
    step = n_step(u_test, total_time)
    print("The whole num_iteration is:", step)