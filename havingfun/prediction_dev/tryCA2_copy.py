import numpy as np
import imageio

# Cell States
# 0 = Clear, 1 = Fuel, 2 = Fire

prob = .6 # probability of a cell being fuel, otherwise it's clear
total_time = 300 # simulation time
terrain_size = [100,100] # size of the simulation: 10000 cells

# states hold the state of each cell
states = np.zeros((total_time,*terrain_size))
# initialize states by creating random fuel and clear cells
states[0] = np.random.choice([0,1],size=terrain_size,p=[1-prob,prob])
# set the middle cell on fire!!!
states[0,terrain_size[0]//2,terrain_size[1]//2] = 2

for t in range(1,total_time):
    # Make a copy of the original states
    states[t] = states[t-1].copy()

    for x in range(1,terrain_size[0]-1):
        for y in range(1,terrain_size[1]-1):

            if states[t-1,x,y] == 2: # It's on fire
                states[t,x,y] = 0 # Put it out and clear it
                
                # If there's fuel surrounding it
                # set it on fire!
                if states[t-1,x+1,y] == 1: 
                    states[t,x+1,y] = 2
                if states[t-1,x-1,y] == 1:
                    states[t,x-1,y] = 2
                if states[t-1,x,y+1] == 1:
                    states[t,x,y+1] = 2
                if states[t-1,x,y-1] == 1:
                    states[t,x,y-1] = 2

colored = np.zeros((total_time,*terrain_size,3),dtype=np.uint8)

# Color
for t in range(states.shape[0]):
    for x in range(states[t].shape[0]):
        for y in range(states[t].shape[1]):
            value = states[t,x,y].copy()

            if value == 0:
                colored[t,x,y] = [139,69,19] # Clear
            elif value == 1: 
                colored[t,x,y] = [0,255,0]   # Fuel
            elif value == 2: 
                colored[t,x,y] = [255,0,0]   # Burning
            
# Crop
cropped = colored[:300,1:terrain_size[0]-1,1:terrain_size[1]-1]

imageio.mimsave('./predict001.gif', cropped)