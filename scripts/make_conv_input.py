import numpy as np
import sys
import torch

IC = int(sys.argv[1])
OC = int(sys.argv[2])
ID = int(sys.argv[3])

fil = np.load(sys.argv[4])
bias= None
if len(sys.argv) > 5:
    bias = np.load(sys.argv[5])

residual = None
if len(sys.argv) > 6 and int(sys.argv[6]) == 1:
    residual = np.random.normal(size=(OC,ID,ID))
    #residual = np.zeros(shape=(A,int(sys.argv[2])))
    np.save("residual.npy",residual.astype(np.float32))

assert fil.shape[0] == OC and fil.shape[1] == IC * 9
np.save("filter.npy",fil)
if bias:
    np.save("bias.npy",bias)
np.save("transposed_filter.npy",fil.transpose().copy())
fil_KFFC = fil.reshape([OC,IC,3,3]).transpose([0,2,3,1]).copy()
np.save("filter_KFFC.npy",fil_KFFC)

im = np.random.normal(size=(IC,ID,ID)).astype(np.float32)
np.save("input.npy",im)

fil = torch.Tensor(fil.reshape([OC,IC,3,3]))
im = torch.Tensor(im).unsqueeze(0)
if bias:
    result = torch.nn.functional.conv2d(im,fil,bias=torch.Tensor(bias),padding=1)
else:
    result = torch.nn.functional.conv2d(im,fil,padding=1)

if residual is not None:
    result = torch.nn.functional.relu(torch.add(result,torch.Tensor(residual)))
    np.save("reference.npy",result.numpy())
else:
    result = torch.nn.functional.relu(result)
    np.save("reference.npy",result.numpy())
