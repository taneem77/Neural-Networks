/*
 this is basically a basic file : 
 youve already leanrt the weights and bias which is loaded into kernel and then u find z which gives u 1 or 0 so the training right now is happening in python but demonstrated in kernel
 */
// also files are loaded into kernel using.ko 
// these are linux kernel headers
#include <linux/init.h> // __init and __exit
#include <linux/kernel.h>//kernel logging
#include <linux/module.h>//functionality requred by loadable kernel modules

#include "perceptron_model.h"

MODULE_LICENSE("GPL"); //metadata
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
{//gets features as input data and then activation to store z 
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
static void run_demo_inference(void) //statis cus it has internal linker which is helping it 
{
    /*
     * Example standardized input:
     *
     * [0.5, -0.2, 1.1, 0.8]
     *
     * represented with SCALE_FACTOR = 1000 bro this is basically ki ull need to specify ki kaunsa kitna decimals u wanna go 
     its important cus heare u arent using standardscaler
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

    pr_info("perceptron: activation = %ld\n", activation); //z value

    pr_info("perceptron: prediction = %d\n", prediction); // is 0 or 1 true or false 
}


/*
 * Called when the module is inserted into the kernel.
 */
static int __init perceptron_init(void)
{
    pr_info("perceptron: kernel module loading\n"); //kernel code mein u wanna print so 

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

// ── Perceptron Kernel Architecture ───────────────────────────────────────────
//
//                     USER SPACE
// ─────────────────────────────────────────────────────────────────────────────
//
//              Python Perceptron
//                     ↓
//                  TRAINING
//                     ↓
//            learned weights + bias
//                     ↓
//         convert to fixed-point integers
//                     ↓
//           perceptron_model.h
//
//
//                    KERNEL SPACE
// ─────────────────────────────────────────────────────────────────────────────
//
//           perceptron_kernel.c
//                     ↓
//              module loaded
//                     ↓
//            perceptron_init()
//                     ↓
//           run_demo_inference()
//                     ↓
//         hardcoded scaled sample
//                     ↓
//           perceptron_predict()
//                     ↓
//         ┌──────────────────────┐
//         │     z = w·x + b      │
//         └──────────────────────┘
//                     ↓
//              is z >= 0?
//                /        \
//              yes         no
//               ↓           ↓
//               1           0
//                     ↓
//           pr_info() → kernel log
//
//
//              module removed
//                     ↓
//           perceptron_exit()
//
// ─────────────────────────────────────────────────────────────────────────────
// Training happens in user space.
// Kernel space only performs inference using the already-trained weights
// and bias stored in perceptron_model.h.
// ─────────────────────────────────────────────────────────────────────────────
