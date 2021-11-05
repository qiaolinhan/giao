# a grid of 100x00 cells
# each cell can be in 3 (5) states: clear, fuel, `heating`, buruning, `ash`
# if cell is burning, it will ignite the heating around it.
import numpy as np
import imageio

# cell states
# 0 = clear, 1 = fuel, 2 = heating, burning = 3, ash = 4
# probability of a cell can being fuel, or it is clear
prob_f = 0.7
p = [prob_f, 1-prob_f]
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
states[0, terrian_size[0]//2, terrian_size[1]//2] = 3

for t in range(1, total_time-1):
    states[t] = states[t-1].copy()

    for x in range(1, terrian_size[0]-1):
        for y in range(1, terrian_size[1]-1):
            # if it is burning, put it out into ash
            if states[t-1, x, y] == 3:
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

            if states[t-1, x, y] == 2:
                states[t, x, y] == 3

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

croped = colored[:200, 1:terrian_size[0]-1, 1:terrian_size[1]-1]

imageio.mimsave('./predict.gif', croped)


# def ca():
#     # celluar automata
#     # list representing the current status of 64 cells
#    ca = [
#          0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,
#          0,0,0,0,0,0,0,0,0,0, 0,1,0,0,0,0,0,0,0,0,
#          0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,  0,0,0,0]

#     ca_new = ca[:]

#     dic = {0: 'F', 1: 'T'}

#     #initial draw, step 0
#     print ''.join(dic[e] for e in ca_new)
#     step = 1
#     while(step <32):
#         ca_new = []
#         for i in range(0, 64):
#             if i>0 and i<63:
#                 if ca[i-1] == ca[i+1]:
#                     ca_new.append(0)
#                 else:
#                     ca_new.append(1)

#             elif(i==0):
#                 if ca[1] == 1:
#                     ca_new.append(1)
#                 else:
#                     ca_new.append(0)

#             elif(i==63):
#                 if ca[62] ==1:
#                     ca_new.append(1)
#                 else:
#                     ca_new.append(0)

#     # draw current cell state
#     print ''.join([dic[e] for e in ca_new])

#     ca - ca_new[:]

#     step += 1
