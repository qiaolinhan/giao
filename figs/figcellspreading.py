from os import readlink
from typing_extensions import Concatenate
import pygraphviz as pgv

G = pgv.AGraph(directed=True, strict=False, ranksep = 0.2, 
               spline = 'ortho', rankdir = 'LR', Concatenate=True)

a0 = 's(x, y, t) = 0'
a1 = 's(x, y, t) = 1'
a2 = 's(x, y, t) = 2'
a3 = 's(x, y, t) = 3'

a00 = 's(x+-1, y+-1, t)0 = 0'
a01 = 's(x+-1, y+-1, t)0 = 1'
a02 = 's(x+-1, y+-1, t)0 = 2'
a03 = 's(x+-1, y+-1, t)0 = 3'

a10 = 's(x+-1, y+-1, t)1 = 0'
a11 = 's(x+-1, y+-1, t)1 = 1'
a12 = 's(x+-1, y+-1, t)1 = 2'
a13 = 's(x+-1, y+-1, t)1 = 3'

a20 = 's(x+-1, y+-1, t)2 = 0'
a21 = 's(x+-1, y+-1, t)2 = 1'
a22 = 's(x+-1, y+-1, t)2 = 2'
a23 = 's(x+-1, y+-1, t)2 = 3'

a30 = 's(x+-1, y+-1, t)3 = 0'
a31 = 's(x+-1, y+-1, t)3 = 1'
a32 = 's(x+-1, y+-1, t)3 = 2'
a33 = 's(x+-1, y+-1, t)3 = 3'

a000 = 's(x, y, t+1) = 0, s(x+-1, y+-1, t+1) = 0'
a010 = 's(x, y, t+1) = 0, s(x+-1, y+-1, t+1) = 1'
a020 = 's(x, y, t+1) = 0, s(x+-1, y+-1, t+1) = 2'
a030 = 's(x, y, t+1) = 0, s(x+-1, y+-1, t+1) = 3'

a100 = 's(x, y, t+1) = 1, s(x+-1, y+-1, t+1) = 0'
a110 = 's(x, y, t+1) = 1, s(x+-1, y+-1, t+1) = 1'
a120 = 's(x, y, t+1) = 2, s(x+-1, y+-1, t+1) = 3'
a130 = 's(x, y, t+1) = 2, s(x+-1, y+-1, t+1) = 3'

a200 = 's(x, y, t+1) = 3, s(x+-1, y+-1, t+1) = 0'
a210 = 's(x, y, t+1) = 3, s(x+-1, y+-1, t+1) = 2'
a220 = 's(x, y, t+1) = 3, s(x+-1, y+-1, t+1) = 3'
a230 = 's(x, y, t+1) = 3, s(x+-1, y+-1, t+1) = 3'

a300 = 's(x, y, t+1) = 3, s(x+-1, y+-1, t+1) = 0'
a310 = 's(x, y, t+1) = 3, s(x+-1, y+-1, t+1) = 2'
a320 = 's(x, y, t+1) = 3, s(x+-1, y+-1, t+1) = 3'
a330 = 's(x, y, t+1) = 3, s(x+-1, y+-1, t+1) = 3'

G.add_nodes_from([a0, a1, a2, a3, 
                  a00, a01, a02, a03,
                  a10, a11, a12, a13,
                  a20, a21, a22, a23,
                  a30, a31, a32, a33,
                  a000, a010, a020, a030, a100, a110, a120, a130,
                  a200, a210, a220, a230, a300, a310, a320, a330,],
                shape = 'polygen', style = 'solid')

G.add_edges_from([[a0, a00], [a0, a01], [a0, a02], [a0, a03],
                  [a1, a10], [a1, a11], [a1, a12], [a1, a13],
                  [a2, a20], [a2, a21], [a2, a22], [a2, a23],
                  [a3, a30], [a3, a31], [a3, a32], [a3, a33],
                  [a00, a000], [a01, a010], 
                  [a02, a020], [a03, a030],
                  [a10, a100], [a11, a110],
                  [a12, a120], [a13, a130],
                  [a20, a200], [a21, a210],
                  [a22, a220], [a23, a230],
                  [a30, a300], [a31, a310],
                  [a32, a320], [a33, a330],],
                   arrowsize = 0.8)

G.layout()
G.draw('figcellspreading.png', prog = 'dot')