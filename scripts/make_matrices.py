import numpy as np
import sys

A_dim = int(sys.argv[1])
B_dim = int(sys.argv[2])
C_dim = int(sys.argv[3])

AB = np.random.normal(size=(A_dim,B_dim)).astype(np.float32)
BC = np.random.normal(size=(B_dim,C_dim)).astype(np.float32)

np.save("AB.npy",AB)
np.save("BC.npy",BC)
np.save("ref.npy",np.dot(AB,BC))
