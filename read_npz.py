import numpy as np

import sys

name = sys.argv[1]

b = np.load(name)

print(b["plddt"])

print(b["plddt"].shape)
