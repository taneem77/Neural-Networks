#ifndef PERCEPTRON_KMOD_H
#define PERCEPTRON_KMOD_H

#include <linux/ioctl.h>

/*
 * Maximum vector length supported.
 */
#define MAX_VEC_LEN 64

/*
 * Data exchanged between user space and kernel space.
 *
 * Inputs:
 *      inputs[]  -> feature vector
 *      weights[] -> model weights
 *      len       -> number of valid elements
 *
 * Output:
 *      result    -> computed dot product
 */
struct dot_product_request
{
    long inputs[MAX_VEC_LEN];
    long weights[MAX_VEC_LEN];
    int len;
    long result;
};

/*
 * ioctl definitions
 */

#define PERCEPTRON_MAGIC 'P'

#define PERCEPTRON_DOT \
        _IOWR(PERCEPTRON_MAGIC, 1, struct dot_product_request)

#endif
