import pygraphviz as pgv

G = pgv.AGraph(directed=True, strict=False, ranksep=0.2, 
               splines="spline", rankdir = "TB", concentrate=True)

a0 = "Dataset"
a1 = "Deep wise: NxN Conv * in_channels"

a2 = "Point wise: 1x1 Conv"
a3 = "Original feature map"

G.add_node(a0, shape = "cylinder", style = "solid")
G.add_nodes_from([a1, a2, a3],
                  shape = "polygen", style = "solid")



G.add_edges_from([[a0, a1], [a1, a2], [a2, a3]],
                  arrowsize=0.8)
                  

G.layout()
G.draw("figuredepthwise.png", prog="dot")




