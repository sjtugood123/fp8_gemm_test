from timeit import default_timer

import pandas

from fprev import AccumImpl, fprev, basic_fprev
from fprev_torch import TorchGEMM, TorchF16GEMM
from graphviz import Digraph

n=128
# impl = TorchGEMM(n, True)
impl = TorchF16GEMM(n,True)

# tree = basic_fprev(impl)
tree = fprev(impl)

# 保存并在本地打开 (会生成 AccumTree.gv.pdf)
tree.render(f"AccumTree{n}", format='png')
print(f"计算树已生成：AccumTree{n}.png")