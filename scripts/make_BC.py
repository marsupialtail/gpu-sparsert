import numpy as np
import sys
matrix = np.load(sys.argv[1])
A = matrix.shape[1]
B = matrix.shape[0]
a = np.random.normal(size=(B,int(sys.argv[2])))

if len(sys.argv) > 4:
    in_format = sys.argv[3]
    out_format = sys.argv[4]
else:
    in_format = "NCHW"
    out_format = "NCHW"

fuse = False
if len(sys.argv) > 5:
    bias = np.load(sys.argv[5])
    fuse = True

residual = None
if len(sys.argv) > 6 and int(sys.argv[6]) == 1:
    residual = np.random.normal(size=(A,int(sys.argv[2])))
    #residual = np.zeros(shape=(A,int(sys.argv[2])))
    np.save("residual.npy",residual.astype(np.float32))

no_relu = False
if len(sys.argv) > 7:
    no_relu = True

print(in_format,out_format)

if in_format == "NCHW":
    np.save("BC.npy",a.astype(np.float32))
elif in_format == "NHWC":
    np.save("BC.npy",a.astype(np.float32).transpose().copy())
else:
    print("Unsupported in format")

if residual is None:
    if not fuse:
        if out_format == "NCHW":
            np.save("ref.npy",np.dot(matrix.transpose(),a).astype(np.float32))
        elif out_format == "NHWC":
            np.save("ref.npy",np.dot(matrix.transpose(),a).astype(np.float32).transpose().copy())
        else:
            print("Unsupported out format")
    else:
        if out_format == "NCHW":
            if no_relu:
                np.save("ref.npy",(np.dot(matrix.transpose(),a) + np.expand_dims(bias,1)).astype(np.float32))
            else:
                np.save("ref.npy",np.maximum(0,np.dot(matrix.transpose(),a) + np.expand_dims(bias,1)).astype(np.float32))
        elif out_format == "NHWC":
            np.save("ref.npy",np.maximum(0,np.dot(matrix.transpose(),a) + bias).astype(np.float32).transpose().copy())
        else:
            print("Unsupported out format")
else:
    if not fuse:
        if out_format == "NCHW":
            np.save("ref.npy",(np.dot(matrix.transpose(),a) + residual).astype(np.float32))
        elif out_format == "NHWC":
            np.save("ref.npy",(np.dot(matrix.transpose(),a).astype(np.float32) + residual).transpose().copy())
        else:
            print("Unsupported out format")
    else:
        if out_format == "NCHW":
            #np.save("ref.npy",(np.maximum(0,np.dot(matrix.transpose(),a) + np.expand_dims(bias,1)) + residual).astype(np.float32))
            np.save("ref.npy",(np.dot(matrix.transpose(),a) + np.expand_dims(bias,1) + residual).astype(np.float32))
        elif out_format == "NHWC":
            np.save("ref.npy",(np.maximum(0,np.dot(matrix.transpose(),a) + bias) + residual).astype(np.float32).transpose().copy())
        else:
            print("Unsupported out format")
