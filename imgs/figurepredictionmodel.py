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

a0 = "Numerical Wildfire Spreading Predictors"

a1 = "Pysical and Chemical Phynomena-based Models"
a2 = "Mathematical Models"

a11 = "Computational-fluid-dynamics (CFD) Models"
a12 = "..."

a21 = "Fire Area Simulator (FARSITE)"
a22 = "Grid-based Models "
a23 = "Continuous Plane-based Models"
a24 = "Fuzzy/Neural Models"
a25 = "... "

a211 = "Kalman Filter and Atmosphere-coupled"
a212 = "Fire Fronts Meansurement-based"
a213 = "Global Optimization Enhanced"
a214 = "...  "

a221 = "Celluar Automata"
a222 = "Bond-percolation Method"
a223 = "...   "



G.add_nodes_from([a0, a1, a2, a11, a12, 
                  a21, a22, a23, a24, a25,
                  a211, a212, a213, a214,
                  a221, a222, a223],
                  shape = "polygen", style = "solid")

G.add_edges_from([[a0, a1], [a0, a2],
                  [a1, a11], [a1, a12],
                  [a2, a21], [a2, a22], [a2, a23], [a2, a24], [a2, a25],
                  [a21, a211],[a21, a212], [a21, a213], [a21, a214],
                  [a22, a221], [a22, a222], [a22, a223]],
                  arrowsize=0.8)

G.layout()
G.draw("figurepredictionmodel.png", prog="dot")





