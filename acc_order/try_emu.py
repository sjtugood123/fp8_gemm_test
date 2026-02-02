from timeit import default_timer

import pandas

from fprev import AccumImpl, fprev, basic_fprev
from experiments.accumimpls.torch import TorchGEMM
from graphviz import Digraph

n=128
impl = TorchGEMM(n, True)

tree = basic_fprev(impl)

# 保存并在本地打开 (会生成 AccumTree.gv.pdf)
tree.render(f"AccumTree{n}", format='png')
print(f"计算树已生成：AccumTree{n}.png")