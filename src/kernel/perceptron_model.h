#ifndef PERCEPTRON_KMOD_H
#define PERCEPTRON_KMOD_H

#include <linux/ioctl.h>

#define MAX_VEC_LEN 64

/*
 * Fixed-point (Q16.16) request/response struct.
 * Kernel space has no easy FPU access, so inputs/weights/result
 * are all scaled integers: real_value = raw / 65536.
 */
struct dot_product_request {
    long inputs[MAX_VEC_LEN];
    long weights[MAX_VEC_LEN];
    int len;
    long result;
};

#define PERCEPTRON_MAGIC 'P'
#define PERCEPTRON_DOT _IOWR(PERCEPTRON_MAGIC, 1, struct dot_product_request)

#endif