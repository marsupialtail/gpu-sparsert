import numpy as np
import sys
import torch


depthwise_filters = torch.Tensor(np.load(sys.argv[1])).permute(3,0,1,2)
groupwise_filters = torch.Tensor(np.load(sys.argv[2]).transpose()).unsqueeze(1).unsqueeze(2).permute(0,3,1,2)
depthwise_bias = torch.Tensor(np.load(sys.argv[3]))
groupwise_bias = torch.Tensor(np.load(sys.argv[4]))

input_size = int(sys.argv[5])
pad_in_left = int(sys.argv[6])
pad_in_right = int(sys.argv[7])
pad_out_left = int(sys.argv[8])
pad_out_right = int(sys.argv[9])
stride = int(sys.argv[10])


OC = groupwise_filters.shape[0]
IC = groupwise_filters.shape[1]

input_image = np.random.normal(size=(1,IC,input_size,input_size))
#input_image = np.ones((1,IC,input_size,input_size))
np.save("input_image.npy",input_image.squeeze().astype(np.float32))
input_image = torch.Tensor(input_image)
intermediate = torch.nn.functional.conv2d(input_image, depthwise_filters, bias=depthwise_bias, stride=stride, padding=1, dilation=1, groups=IC)
intermediate = torch.nn.functional.relu(intermediate)
result_1 = torch.nn.functional.conv2d(intermediate,groupwise_filters,bias=groupwise_bias,stride=1,padding=0,dilation=1,groups=1)
result_1 = torch.nn.functional.relu(result_1)
padded_in = torch.nn.functional.pad(input_image,(pad_in_left,pad_in_right,1,1))
np.save("padded_input_image.npy",padded_in.squeeze().data.numpy())
padded_out = torch.nn.functional.pad(result_1,(pad_out_left,pad_out_right,1,1))
np.save("padded_result.npy",padded_out.squeeze().data.numpy())

