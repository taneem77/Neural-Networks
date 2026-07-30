#ifndef PERCEPTRON_MODEL_H //include guard, if perceptron model has not already been defined
#define PERCEPTRON_MODEL_H  
/*mark as define now */

/*
 * perceptron_model.h
 *
 * Fixed-point model parameters for the Linux kernel
 * perceptron inference prototype.
 *
 * SCALE_FACTOR = 1000
 *
 * Examples:
 *   1.0   ->  1000
 *   0.5   ->   500
 *  -0.75  ->  -750
 */

#define NUM_FEATURES 4
#define SCALE_FACTOR 1000

/*
 * Prototype perceptron weights.
 *
 * These values are currently used to test kernel-space
 * inference. They will later be replaced by parameters
 * exported from the trained Python perceptron.
 */
static const long perceptron_weights[NUM_FEATURES] = {
    -500,
     800,
    -1200,
    -900
};

/*
 * Prototype bias.
 */
static const long perceptron_bias = 100;

#endif /* PERCEPTRON_MODEL_H */