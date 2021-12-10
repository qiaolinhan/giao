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
               splines="spline", rankdir = "TB", concentrate=True)

a0 = "Simulated Fire Fronts"

a1 = "ROS Measurement"

a3 = "Cellular Automata"
a4 = "Fire Fronts Prediction"

a2 = "Historical Dataset"

a11 = "Attention-based Regression Model 1"
a21 = "Attention-based Regression Model 2"


G.add_nodes_from([a0, a2], shape = "cylinder", style = "solid")
G.add_nodes_from([a1, a3, a4,
                  a11, a21],
                  shape = "polygen", style = "solid")

G.add_edges_from([[a0, a1], [a1, a3],[a3, a4],
                 [a2, a11], [a11, a1],
                 [a2, a21], [a21, a3]],
                  arrowsize=0.8)

G.layout()
G.draw("figurepredictionprocess.png", prog="dot")