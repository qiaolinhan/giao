'''
Assume width of every cell is 100(m), each cell can be in 4 states: 
    clear, vegetation, ignit, buruning
If cell is burning, it will ignite the heating around it.

cell states
0 = clear, 1 = vegetation, 2 = ignit, 3 = burning
spreading rules: shown in figurecellspreading.py

Assume width of one cell is 100
the step time is \hat t = 100/u_flat or \hat t = 100/u_slope
'''

import numpy as np
import imageio
import math
from eqPoability import prob
from eqRateOfSpreading import ros, n_step

# # measurements
# V = 200
# theta = math.radians(30)
# s = math.radians(15)
# # simulation envs
# total_time = 300
# cell_size = 100
# u_test = ros(V, theta, s)

# probability of ignition, count of spreading steps
P_ignit = 0.8 
# P_ignit = prob(P_veg, P_den, P_w, P_s)
print("======> The probability of ignition is:", P_ignit)

steps = 200
# steps = n_step(u_test, total_time, cell_size)
# print("======> The rate of spreading is:", u_test)
print("======> The whole num_iteration is:", steps)
p = [P_ignit, 1-P_ignit]

# simulation size
terrian_size = [200, 200]
fild = np.random.choice([1, 0], size = terrian_size, p = p)
# states hold the sate of each cell
states = np.zeros((*terrian_size, steps))
# initialize states
states[0] = fild
# the pm
pm = np.array([+1, -1])

# set the middle cell on fire
states[terrian_size[0]//2, terrian_size[1]//2, 0] = 2

for t in range(1, steps-1):
    states[t] = states[t-1].copy()

    for x in range(1, terrian_size[0]-1):
        for y in range(1, terrian_size[1]-1):
            # if it is burning, it continues
            if states[x, y, t-1].any() >= 2:
                states[x, y, t] == 3
                # igniting neibour cells
                if states[x + pm, y + pm, t-1] == 0:
                    states[x+ pm, y + pm, t] == 0
                elif states[x + pm, y + pm, t-1] == 1:
                    states[x + pm, y + pm, t] == 2
                elif states[x + pm, y + pm, t-1] == 2:
                    states[x + pm, y + pm, t] == 3
                elif states[x + pm, y + pm, t-1] ==3:
                    states[x + pm, y + pm, t] == 3

            elif states[x, y, t-1] == 1:
                # states[x, y, t] == 1
                if states[x + pm, y + pm, t-1] == 0:
                    states[x, y, t] == 1
                elif states[x + pm, y + pm, t-1] == 1:
                    states[x, y, t] == 1
                elif states[x + pm, y + pm, t-1] == 2:
                    states[x, y, t] ==2
                elif states[x + pm, y + pm, t-1] == 3:
                    states[x, y, t] ==2

            elif states[x, y, t-1] == 0:
                states[x, y, t] == 0 

colored = np.zeros((steps, *terrian_size, 3), dtype = np.uint8)

for t in range(states.shape[0]):
    for x in range(states[t].shape[0]):
        for y in range(states[t].shape[0]):
            value = states[t, x, y].copy()

            if value == 0:
                colored[t, x, y] = [0, 0, 0]
            elif value ==1:
                colored[t, x, y] = [0, 255, 0]
            elif value == 2:
                colored[t, x, y] = [255, 255, 0]
            elif value == 3:
                colored[t, x, y] = [255, 111, 0]

croped = colored[:300, 1:terrian_size[0]-1, 1:terrian_size[1]-1]

imageio.mimsave('havingfun/prediction_dev/saved_imgs/prediction.gif', croped)