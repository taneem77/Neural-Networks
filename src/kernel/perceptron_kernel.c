/*
 * perceptron_kernel.c
 *
 * Basic Linux kernel implementation of single-layer
 * perceptron inference.
 *
 * This module demonstrates how the mathematical inference
 * operation used by the Python perceptron can be represented
 * inside Linux kernel space using integer arithmetic.
 *
 * Training remains in user space.
 *
 * Author: Tanisha Mathur
 */

#include <linux/init.h>
#include <linux/kernel.h>
#include <linux/module.h>

#include "perceptron_model.h"

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Tanisha Mathur");
MODULE_DESCRIPTION(
    "Single-layer perceptron inference prototype for Linux kernel space"
);
MODULE_VERSION("0.1");


/*
 * perceptron_predict
 *
 * Performs:
 *
 *      z = w1*x1 + w2*x2 + ... + wn*xn + bias
 *
 * followed by:
 *
 *      prediction = 1 if z >= 0
 *                   0 otherwise
 *
 * Features are expected to already be represented using the
 * same fixed-point scale as the model.
 */
static int perceptron_predict(const long features[NUM_FEATURES],
                              long *activation)
{
    int i;
    long weighted_sum = perceptron_bias * SCALE_FACTOR;

    for (i = 0; i < NUM_FEATURES; i++) {
        weighted_sum += perceptron_weights[i] * features[i];
    }

    *activation = weighted_sum;

    return weighted_sum >= 0 ? 1 : 0;
}


/*
 * run_demo_inference
 *
 * Runs one sample through the kernel-space perceptron when
 * the module is loaded.
 *
 * This is intentionally a minimal prototype. A later stage
 * will receive samples dynamically from user space.
 */
static void run_demo_inference(void)
{
    /*
     * Example standardized input:
     *
     * [0.5, -0.2, 1.1, 0.8]
     *
     * represented with SCALE_FACTOR = 1000.
     */
    long sample[NUM_FEATURES] = {
         500,
        -200,
        1100,
         800
    };

    long activation;
    int prediction;

    prediction = perceptron_predict(sample, &activation);

    pr_info("perceptron: demo input = [%ld, %ld, %ld, %ld]\n",
            sample[0],
            sample[1],
            sample[2],
            sample[3]);

    pr_info("perceptron: activation = %ld\n", activation);

    pr_info("perceptron: prediction = %d\n", prediction);
}


/*
 * Called when the module is inserted into the kernel.
 */
static int __init perceptron_init(void)
{
    pr_info("perceptron: kernel module loading\n");

    pr_info("perceptron: model contains %d input features\n",
            NUM_FEATURES);

    pr_info("perceptron: fixed-point scale = %d\n",
            SCALE_FACTOR);

    run_demo_inference();

    pr_info("perceptron: module loaded successfully\n");

    return 0;
}


/*
 * Called when the module is removed from the kernel.
 */
static void __exit perceptron_exit(void)
{
    pr_info("perceptron: kernel module unloaded\n");
}


module_init(perceptron_init);
module_exit(perceptron_exit);