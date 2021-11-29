from os import readlink
from typing_extensions import Concatenate
import pygraphviz as pgv

G = pgv.AGraph(directed=True, strict=False, ranksep = 0.2, 
               spline = 'spline', rankdir = 'LR', Concatenate=True)

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

a30 = 's(x+-1, y+-1, t)0 = 0'
a31 = 's(x+-1, y+-1, t)0 = 1'
a32 = 's(x+-1, y+-1, t)0 = 2'
a33 = 's(x+-1, y+-1, t)0 = 3'

a000 = 's(x, y, t+1) = 0'
a001 = 's(x+-1, y+-1, t+1) = 0'
a010 = 's(x, y, t+1) = 0'
a011 = 's(x+-1, y+-1, t+1) = 1'
a020 = 's(x, y, t+1) = 0'
a021 = 's(x+-1, y+-1, t+1) = 2'
a030 = 's(x, y, t+1) = 0'
a031 = 's(x+-1, y+-1, t+1) = 3'

a100 = 's(x, y, t+1) = 1'
a101 = 's(x+-1, y+-1, t+1) = 0'
a110 = 's(x, y, t+1) = 1'
a111 = 's(x+-1, y+-1, t+1) = 1'
a120 = 's(x, y, t+1) = 2'
a121 = 's(x+-1, y+-1, t+1) = 3'
a130 = 's(x, y, t+1) = 2'
a131 = 's(x+-1, y+-1, t+1) = 3'

a200 = 's(x, y, t+1) = 3'
a201 = 's(x+-1, y+-1, t+1) = 0'
a210 = 's(x, y, t+1) = 3'
a211 = 's(x+-1, y+-1, t+1) = 2'
a220 = 's(x, y, t+1) = 3'
a221 = 's(x+-1, y+-1, t+1) = 3'
a230 = 's(x, y, t+1) = 3'
a231 = 's(x+-1, y+-1, t+1) = 3'

a300 = 's(x, y, t+1) = 3'
a301 = 's(x+-1, y+-1, t+1) = 0'
a310 = 's(x, y, t+1) = 3'
a311 = 's(x+-1, y+-1, t+1) = 2'
a320 = 's(x, y, t+1) = 3'
a321 = 's(x+-1, y+-1, t+1) = 3'
a330 = 's(x, y, t+1) = 3'
a331 = 's(x+-1, y+-1, t+1) = 3'

G.add_nodes_from([a0, a1, a2, a3, 
                  a00, a01, a02, a03,
                  a10, a11, a12, a13,
                  a20, a21, a22, a23,
                  a30, a31, a32, a33,
                  a000, a001, a010, a011, a020, a021, a030, a031,
                  a100, a101, a110, a111, a120, a121, a130, a131,
                  a200, a201, a210, a211, a220, a221, a230, a231,
                  a300, a301, a310, a311, a320, a321, a330, a331],
                shape = 'polygen', style = 'solid')

G.add_edges_from([[a0, a00], [a0, a01], [a0, a02], [a0, a03],
                  [a1, a10], [a1, a11], [a1, a12], [a1, a13],
                  [a2, a20], [a2, a21], [a2, a22], [a2, a23],
                  [a3, a30], [a3, a31], [a3, a32], [a3, a33],
                  [a00, a000], [a00, a001], [a01, a010], [a01, a011],
                  [a02, a020], [a02, a021], [a03, a030], [a03, a031],
                  [a10, a100], [a10, a101], [a11, a110], [a11, a111],
                  [a12, a120], [a12, a121], [a13, a130], [a13, a131],
                  [a20, a200], [a20, a201], [a21, a210], [a21, a211],
                  [a22, a220], [a22, a221], [a23, a230], [a23, a231],
                  [a30, a300], [a30, a301], [a31, a310], [a31, a311],
                  [a32, a320], [a32, a321], [a33, a330], [a33, a331]],
                   arrowsize = 0.2)

G.layout()
G.draw('figcellspreading.png', prog = 'dot')