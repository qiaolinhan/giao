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

a0 = "Feature Map"
a11 = "1x1 Conv"
a12 = "3x3 Conv"
a2 = "Concatenates, ReLU"
a3 = "1x1 Conv, ReLU"
a4 = "Re-sampling"
a5 = "3x3 Conv, ReLU"

G.add_node(a0, shape = "cylinder", style = "solid")
G.add_nodes_from([a11, a12, a2, a3, a4, a5],
                  shape = "polygen", style = "solid")

G.add_edges_from([[a0, a11], [a0, a12]], arrowsize = 0.8)
G.add_edges_from([[a11, a2], [a12, a2]], arrowsize = 0.8)
G.add_edges_from([[a2, a3], [a3, a4], [a4, a5]], arrowsize = 0.8)



G.layout()
G.draw("figureunsqueezenetflow.png", prog="dot")