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

a0 = "Visual Object Tracking (VOT) Models"

a1 = "Generative Models"
a2 = "Discriminative Models"

a11 = "Kalman Filter/Partical Filter-based"
a12 = "Mean-shift/Cam-Shift"
a13 = "Optical Flow"
a14 = "..."

a21 = "Machine Learning Classifiers"
a22 = "Corelation Filter"
a23 = "... "

a211 = "Support Vevtor Machine (SVM)"
a212 = "Random Forest"
a213 = "Deep ConvNet"
a214 = "...  "

G.add_nodes_from([a0, 
                  a1, a2, 
                  a11, a12, a13, a14,
                  a21, a22, a23,
                  a211, a212, a213, a214,],
                  shape = "polygen", style = "solid")

G.add_edges_from([[a0, a1], [a0, a2],
                  [a1, a11], [a1, a12], [a1, a13], [a1, a14],
                  [a2, a21], [a2, a22], [a2, a23],
                  [a21, a211],[a21, a212], [a21, a213], [a21, a214]],
                  arrowsize=0.8)

G.layout()
G.draw("figuretrackingmodel.png", prog="dot")




