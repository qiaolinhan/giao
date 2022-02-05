import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(0, '/home/qiao/dev/giao/havingfun/deving/common')

from inout import Conv0, Conv_f
from attentiongate import Attention_block
from doubleconv import Block, TBlock