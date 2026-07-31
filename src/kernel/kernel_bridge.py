"""
kernel_bridge.py
kernel_bridge.py acts as the communication layer between Python and the Linux kernel. It packages the input and weight vectors into the same format expected by the kernel module, sends them using an ioctl() call, receives the computed dot product, converts it back to floating-point, and returns it to the Python application

Calls the perceptron_kmod char device to compute a weighted sum
(dot product) in kernel space, using Q16.16 fixed-point ioctl.

Drop this alongside your existing day3/day4 perceptron files and
swap the numpy dot product call for kernel_dot_product() to show
the kernel-accelerated path in your CLI/demo.
"""

import ctypes #python cant understand the c files and requests
import fcntl #gives accest to ioctl i.e special req sent to kernel 
import os #lets python talk to os 

MAX_VEC_LEN = 64 #max length of characters it can take 
Q_SHIFT = 16 #decimal precision

class DotProductRequest(ctypes.Structure):
    _fields_ = [
        ("inputs", ctypes.c_long * MAX_VEC_LEN),
        ("weights", ctypes.c_long * MAX_VEC_LEN),
        ("len", ctypes.c_int),
        ("result", ctypes.c_long),
    ] #this whole thing is like a blueprint of the order od data linux requires

_PERCEPTRON_MAGIC = ord('P')

def _iowr(magic, nr, size):
    """Reimplements the Linux _IOWR() macro so we don't need a C
    extension just to get the ioctl command number."""
    IOC_READ_WRITE = (2 << 30) | (1 << 30)  # _IOC_READ | _IOC_WRITE
    return IOC_READ_WRITE | (magic << 8) | nr | (size << 16)

PERCEPTRON_DOT = _iowr(_PERCEPTRON_MAGIC, 1, ctypes.sizeof(DotProductRequest))


def _to_fixed(x: float) -> int: #fixed point arithematic
    return int(round(x * (1 << Q_SHIFT)))


def _from_fixed(x: int) -> float:
    return x / (1 << Q_SHIFT)


def kernel_dot_product(inputs, weights, device="/dev/perceptron_kmod"):
    """Computes sum(inputs[i] * weights[i]) via the kernel module."""
    if len(inputs) != len(weights):
        raise ValueError("inputs and weights must be the same length")
    if len(inputs) > MAX_VEC_LEN:
        raise ValueError(f"vector length exceeds MAX_VEC_LEN={MAX_VEC_LEN}")

    req = DotProductRequest() #empty parcel
    for i, (a, b) in enumerate(zip(inputs, weights)):
        req.inputs[i] = _to_fixed(a)
        req.weights[i] = _to_fixed(b)
    req.len = len(inputs)

    fd = os.open(device, os.O_RDWR)
    try:
        fcntl.ioctl(fd, PERCEPTRON_DOT, req)
    finally:
        os.close(fd)

    return _from_fixed(req.result)


if __name__ == "__main__":
    x = [0.5, -0.2, 0.9]
    w = [1.0, 2.0, -1.0]
    print("kernel-computed dot product:", kernel_dot_product(x, w))
    print("expected (python):", sum(a * b for a, b in zip(x, w)))
