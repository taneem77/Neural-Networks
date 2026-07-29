# Linux Kernel Perceptron Prototype

This directory contains the first systems-level extension of the
single-layer perceptron project.

The objective is to investigate how lightweight neural-network inference
can be represented and executed within Linux kernel space.

## Current Scope

The current implementation is intentionally minimal.

Training, preprocessing, evaluation, and model persistence remain in the
existing Python application.

The kernel component currently implements only the inference operation:

    z = w · x + b

followed by a binary step activation:

    prediction = 1 if z >= 0
                 0 otherwise

This separation keeps computationally complex training operations in user
space while exploring lightweight inference at a lower system level.

## Files

### perceptron_kernel.c

Linux loadable kernel module containing:

- module initialization and cleanup
- fixed-point perceptron inference
- weighted-sum calculation
- step activation
- demonstration inference
- kernel logging

### perceptron_model.h

Contains the fixed-point representation of:

- model weights
- model bias
- number of input features
- scaling factor

The current parameters are prototype values.

Automatic export of parameters from the trained Python model is planned as
the next integration step.

### Makefile

Build configuration for compiling the source as an external Linux kernel
module using Kbuild.

## Why Fixed-Point Arithmetic?

The Python implementation uses floating-point NumPy operations.

For the initial kernel prototype, model parameters and features are
represented as scaled integers.

For example, with a scale factor of 1000:

    1.250  ->  1250
    0.500  ->   500
   -0.750  ->  -750

This allows the perceptron's weighted-sum operation to be expressed using
integer arithmetic.

## Architecture

    Python ML Pipeline
            |
            | training
            v
      Learned Model
            |
            | future parameter export
            v
    Fixed-Point Parameters
            |
            v
    Linux Kernel Module
            |
            v
      w.x + b
            |
            v
      Step Activation
            |
            v
        Prediction

## Building

This component must be built on Linux with the development headers for the
currently running kernel installed.

From this directory:

    make

A successful build produces a loadable kernel module:

    perceptron_kernel.ko

## Loading

Load the module:

    sudo insmod perceptron_kernel.ko

Inspect kernel messages:

    sudo dmesg | tail -20

The log should contain messages showing:

- module initialization
- model feature count
- fixed-point scale
- demonstration input
- activation value
- binary prediction

Check whether the module is loaded:

    lsmod | grep perceptron

Remove the module:

    sudo rmmod perceptron_kernel

Inspect the log again:

    sudo dmesg | tail -20

## Current Limitation

The current module performs inference on a demonstration feature vector
defined inside the kernel module.

It does not yet receive live samples from the Python CLI.

The model parameters are also prototype parameters rather than parameters
automatically exported from the trained Python model.

## Next Integration Steps

1. Export trained Python perceptron weights and bias.
2. Quantize the trained parameters into a fixed-point representation.
3. Generate the kernel model header automatically.
4. Add a user-space to kernel-space communication interface.
5. Send standardized Iris samples from the existing CLI.
6. Compare Python and kernel predictions.
7. Benchmark inference latency and interface overhead.

The long-term objective is therefore not to train a neural network inside
the kernel, but to investigate lightweight inference across the
user-space/kernel-space boundary.