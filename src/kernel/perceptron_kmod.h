//its like whenever u communicate this is the format ull use so if kernel expects it in order of weights inputs and result thats what its going to give in

#ifndef PERCEPTRON_KMOD_H
#define PERCEPTRON_KMOD_H

#include <linux/ioctl.h>

/*
 * Maximum vector length supporte
 */
#define MAX_VEC_LEN 64 
//maximum features supported 

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
struct dot_product_request// this will be the package the moves from user space to kernel 
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
//ioctly command

#endif
