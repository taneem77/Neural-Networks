# Kernel-accelerated perceptron dot product

Adds a Linux kernel module (`perceptron_kmod`) that exposes a char
device (`/dev/perceptron_kmod`). Your Python perceptron code sends
input/weight vectors to it via `ioctl`, and the kernel computes the
weighted sum in fixed-point arithmetic (kernel space has no easy
FPU access, so this uses Q16.16 fixed-point instead of floats).

This gives you a genuine kernel-programming component to point to:
char devices, ioctl, copy_from_user/copy_to_user, fixed-point math —
without touching your existing Python architecture.

## Repo layout to add
```
Neural-Networks/
  kernel/
    perceptron_kmod.c
    perceptron_kmod.h
    Makefile
  python/
    kernel_bridge.py   (or wherever your other day-modules live)
```

## Build the module
Requires kernel headers matching your running kernel.

```bash
sudo apt install build-essential linux-headers-$(uname -r)
cd kernel
make
```

This produces `perceptron_kmod.ko`.

If the build fails on `class_create(DEVICE_NAME)` with a
"too many arguments" error, you're on a pre-6.4 kernel — change that
line in `perceptron_kmod.c` to `class_create(THIS_MODULE, DEVICE_NAME)`
and rebuild. The comment above that line in the source notes this.

## Load it
```bash
sudo insmod perceptron_kmod.ko
dmesg | tail        # confirm "perceptron_kmod: loaded, major=..."
ls -l /dev/perceptron_kmod
```

If `/dev/perceptron_kmod` isn't world-accessible, either run your
Python script with sudo, or add a udev rule / chmod it for testing:
```bash
sudo chmod 666 /dev/perceptron_kmod
```

## Run the Python side
```bash
cd ../python
python3 kernel_bridge.py
```
Expected output: the kernel-computed dot product matches the plain
Python computation.

To wire it into your existing perceptron: replace the
`np.dot(inputs, weights)` call in your forward-pass step with
`kernel_dot_product(inputs, weights)` from `kernel_bridge.py`, and
add a CLI flag (e.g. `--backend kernel`) so you can demo both paths
side by side — that's a good demo moment ("here's the same forward
pass running through a kernel module I wrote").

## Unload when done
```bash
sudo rmmod perceptron_kmod
dmesg | tail   # confirm "perceptron_kmod: unloaded"
```

## Resume framing
Describe this as something you built to extend the internship
project with a systems/kernel component — not as an internship
deliverable itself, since it wasn't assigned. A clean line like
"extended the perceptron project with a custom Linux kernel module
(char device + ioctl) to offload the forward-pass dot product to
kernel space" is accurate and reads well for both CSE and ECE-leaning
roles. Keep that distinction ready in case an interviewer asks what
was scoped by the internship vs. self-directed.