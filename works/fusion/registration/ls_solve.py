import  numpy as np

########################
# real distance
d = 5760
 
vi_points = np.array([
            [484, 282],
            [609, 278],
            [425, 403],
            [611, 401]
    ])

ir_points = np.array([
            [374, 301],
            [605, 301],
            [374, 456],
            [606, 455]
    ])

#######################
# construct R' and t'
R = []
t = []

for (x1, y1), (x2, y2) in zip(vi_points, ir_points):
    R.append([x1, y1, 1, 0, 0, 0, 0, 0, 0, -x2*x1, -x2*y1, -x2])
    R.append([0, 0, 0, x1, y1, 1, 0, 0, 0, -y2*x1, -y2*x1, -y2])
    R.append([0, 0, 0, 0, 0, 0, x1, y1, 1, -1*x1, -1*y1, -1])

    t.extend([x2 - 1/d, y2 - 1/d, 1 - 1/d])

R = np.array(R)
t = np.array(t)

# solve for the parameters using least squares
params, _, _, _= np.linalg.lstsq(R, t, rcond = None)

# extract the transformation matrix components
R11, R12, R13, R21, R22, R23, R31, R32, R33, t1, t2, t3 = params

print("Transform matrix:\n", R11, R12, R13, R21, R22, R23, R31, R32, R33)
print("Translation vector (scaled with d):\n", t1, t2, t3)
