'''
G = pgv.AGraph(directed=True, strict=False, nodesep=0, ranksep=1.2, rankdir="TB",
               splines="none", concentrate=True, bgcolor="write",
               compound=True, normalize=False, encoding='UTF-8')

directed -> False | True：有向图
strict -> True | False：简单图
nodesep：同级节点最小间距
ranksep：不同级节点最小间距
rankdir：绘图方向，可选 TB (从上到下), LR (从左到右), BT (从下到上), RL (从右到左)
splines：线条类型，可选 ortho (直角), polyline (折线), spline (曲线), line (线条), none (无)
concentrate -> True | False：合并线条 (双向箭头)
bgcolor：背景颜色
compound -> True | False：多子图时，不允许子图相互覆盖
normalize -> False | True：以第一个节点作为顶节点
'''
import pygraphviz as pgv

G = pgv.AGraph(directed=True, strict=False, ranksep=0.2, 
               splines="spline", rankdir = "LR", concentrate=True)

a0 = "DNN-based Tracker"

a1 = "Feature Learning"
a2 = "Data Association (Multi-object Tracking)"
a3 ="End-to-End Tracking"
a4 = "State Prediction"
a5 = "State Update"

a11 = "Denoising Autoencoder (DAE)"
a12 = "VGG"
a13 = "..."

a21 = "Siamese instance Search Tracker (SINT)"
a22 = "Twin-architecture Siamese Network + Optical Flow"
a23 = "Two-stream DNN"
a24 = "... "

a31 = "Multi-Domain Network (MDNet)"
a32 = "You Only Look Once & Recurrent Neural Network (ROLO)"
a33 = "...  "

a41 = "Behavior-CNN"
a42 = "Social-LSTM"
a43 = "Dynamic Occupancy Grid Map (DOGMa)"
a44 = "...   "

a51 = "Kalman Filter/ Partical Filter"
a52 = "...    "

G.add_nodes_from([a0, 
                  a1, a2, a3, a4, a5,
                  a11, a12, a13,
                  a21, a22, a23, a24,
                  a31, a32, a33,
                  a41, a42, a43, a44,
                  a51, a52],
                  shape = "polygen", style = "solid")

G.add_edges_from([[a0, a1], [a0, a2], [a0, a3], [a0, a4], [a0, a5],
                  [a1, a11], [a1, a12], [a1, a13],
                  [a2, a21], [a2, a22], [a2, a23], [a2, a24],
                  [a3, a31], [a3, a32], [a3, a33],
                  [a4, a41], [a4, a42], [a4, a43], [a4, a44],
                  [a5, a51], [a5, a52]],
                  arrowsize=0.8)

G.layout()
G.draw("figureDNNtrackingmodel.png", prog="dot")




