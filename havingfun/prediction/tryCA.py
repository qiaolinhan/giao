# a grid of 100x00 cells
# each cell can be in 5 states: clear, vegetation, heating, buruning, flamed
# if cell is burning, it will ignite the heating around it.
'''
# cell states
# 0 = clear, 1 = vegetation, 2 = heating, burning = 3, flamed = 4

# rules
# 0 --> 0, 2 --> 3, 3 --> 4, 4 --> 4

# rate of spreading
# if theta < pi/2: u = epsilon + alpha * \sqrt{v_wind * cos^n(theta)}
# if theta > pi/2: u = epsilon * (beta + (1 - beta) * |sin(theta)|)
# if sloppe: u = u * e^{2*slope}
# params: epsilon, alpha, 
# measurement: v_wind,  
'''

import numpy as np
import imageio

P_o = 0.8 # P_original
p = [P_o, 1-P_o]

# simulation time
total_time = 300
# simulation size
terrian_size = [200, 200]
fild = np.random.choice([1, 0], size = terrian_size, p=p)
# states hold the sate of each cell
states = np.zeros((total_time, *terrian_size))
# initialize states
states[0] = fild

# set the middle cell on fire
states[0, terrian_size[0]//2, terrian_size[1]//2] = 2

for t in range(1, total_time-1):
    states[t] = states[t-1].copy()

    for x in range(1, terrian_size[0]-1):
        for y in range(1, terrian_size[1]-1):
            # if it is burning, put it out into ash
            if states[t-1, x, y] == 2:
                states[t, x, y] == 3
                states[t+1, x, y]== 4

                # heating process
                if states[t-1, x+5, y] == 1:
                    states[t, x+5, y] == 2
                if states[t-1, x-5, y] ==1:
                    states[t, x-5, y] == 2
                if states[t-1, x, y+5] == 1:
                    states[t, x, y+5] == 2
                if states[t-1, x, y-5] ==1:
                    states[t, x, y-5] == 2

colored = np.zeros((total_time, *terrian_size, 3), dtype = np.uint8)

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
            elif value == 4:
                colored[t, x, y] = [96, 96, 96]

croped = colored[:300, 1:terrian_size[0]-1, 1:terrian_size[1]-1]

imageio.mimsave('./predict.gif', croped)