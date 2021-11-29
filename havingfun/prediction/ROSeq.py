# probability of a cell can being fuel, or it is clear
# P_burn = P_origial * (1+P_veg(1+P_density)) * P_wind * P_slope
# if |theta| < pi/2: U(V, theta) = epsilon_0 + alpha * \sqrt(V*cos^v(theta))
# if |theta| > pi>2: U(V, theta) = epsilon_0 * (beta + (1-beta)*|sin(theta)|)
# if slope: U(V, theta) = U * e^{2s}
import math

# params
epsilon_0 = 11
alpha = 1
beta = 1
n = 2 # times

# measurements
theta = math.radians(30) # between wind direction and normal spreading direction
V = 200 # m/s
s = math.radians(15) # slope of topography

# rate of spreading
def U(V, theta, s=None):
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

# spreading probability
# P_h: original prob
# P_veg, P_den: vegetation related prob
# P_w, P_s: wind velosity, topography slope ralated prob
# P_burn = P_h * (1 + P_veg * (1 + P_den)) * P_w * P_s
