import sys
import numpy as np

filename = sys.argv[1]
matrix = np.load(filename)
stem = filename.replace("./npy","")
row_name = stem + "_rows.npy"
col_name = stem + "_cols.npy"

row_offsets = []
cols = []

for i in range(matrix.shape[0]):
    row_offsets.append(len(cols))
    for j in range(matrix.shape[1]):
        if np.abs(matrix[i,j]) > 0.00000001:
            cols.append(j)

row_offsets.append(len(cols))
np.save(row_name,np.array(row_offsets).astype(np.int32))
np.save(col_name,np.array(cols).astype(np.int32))
