import os
import struct
import fcntl
import numpy as np

DEVICE="/dev/perceptron_kmod"
IOCTL_RUN=100


def kernel_predict(sample):

    fd=os.open(DEVICE,os.O_RDWR)

    packed=struct.pack("4i",*sample)

    fcntl.ioctl(fd,IOCTL_RUN,packed)

    result=os.read(fd,4)

    os.close(fd)

    return struct.unpack("i",result)[0]


def python_predict(weights,bias,sample):

    scaled=np.array(sample)/10

    z=np.dot(weights,scaled)+bias

    return int(z>=0)


def verify():

    model=np.load("../saved_model.npz")

    weights=model["weights"]
    bias=float(model["bias"][0])

    samples=[
        [51,35,14,2],
        [63,29,47,16],
        [59,30,51,18],
        [50,36,14,2]
    ]

    print("="*40)
    print(" PYTHON ↔ KERNEL VERIFICATION ")
    print("="*40)

    correct=0

    for s in samples:

        py=python_predict(weights,bias,s)

        kr=kernel_predict(s)

        status="PASS" if py==kr else "FAIL"

        if py==kr:
            correct+=1

        print(f"{s}  Python:{py}  Kernel:{kr}  {status}")

    print("-"*40)
    print(f"Matched {correct}/{len(samples)} samples")

    if correct==len(samples):
        print("100% IDENTICAL INFERENCE")
    else:
        print("Kernel mismatch detected")


if __name__=="__main__":
    verify()