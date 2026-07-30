#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/fs.h>
#include <linux/uaccess.h>
#include <linux/cdev.h>
#include <linux/device.h>
#include "perceptron_kmod.h"

#define DEVICE_NAME "perceptron_kmod"

static int major;
static struct class *perceptron_class;
static struct cdev perceptron_cdev;

/* Computes sum(inputs[i] * weights[i]) in Q16.16 fixed point.
 * This is the same math as the perceptron's weighted-sum step in
 * day3/day4 of the Python project, just done in kernel space via
 * ioctl instead of numpy, to demonstrate a kernel-accelerated path.
 */
static long perceptron_ioctl(struct file *file, unsigned int cmd, unsigned long arg)
{
    struct dot_product_request req;
    long acc;
    int i;

    if (cmd != PERCEPTRON_DOT)
        return -ENOTTY;

    if (copy_from_user(&req, (struct dot_product_request __user *)arg, sizeof(req)))
        return -EFAULT;

    if (req.len < 0 || req.len > MAX_VEC_LEN)
        return -EINVAL;

    acc = 0;
    for (i = 0; i < req.len; i++)
        acc += (req.inputs[i] * req.weights[i]) >> 16;

    req.result = acc;

    if (copy_to_user((struct dot_product_request __user *)arg, &req, sizeof(req)))
        return -EFAULT;

    pr_info("perceptron_kmod: dot product over %d elements -> %ld\n", req.len, acc);
    return 0;
}

static const struct file_operations perceptron_fops = {
    .owner = THIS_MODULE,
    .unlocked_ioctl = perceptron_ioctl,
};

static int __init perceptron_init(void)
{
    dev_t dev;
    int ret;

    ret = alloc_chrdev_region(&dev, 0, 1, DEVICE_NAME);
    if (ret < 0)
        return ret;
    major = MAJOR(dev);

    cdev_init(&perceptron_cdev, &perceptron_fops);
    ret = cdev_add(&perceptron_cdev, dev, 1);
    if (ret < 0) {
        unregister_chrdev_region(dev, 1);
        return ret;
    }

    /* NOTE: class_create() dropped its owner/THIS_MODULE argument in
     * kernel 6.4+. If you're on an older kernel and this fails to build,
     * change the line below to: class_create(THIS_MODULE, DEVICE_NAME)
     */
    perceptron_class = class_create(DEVICE_NAME);
    if (IS_ERR(perceptron_class)) {
        cdev_del(&perceptron_cdev);
        unregister_chrdev_region(dev, 1);
        return PTR_ERR(perceptron_class);
    }
    device_create(perceptron_class, NULL, dev, NULL, DEVICE_NAME);

    pr_info("perceptron_kmod: loaded, major=%d\n", major);
    return 0;
}

static void __exit perceptron_exit(void)
{
    dev_t dev = MKDEV(major, 0);

    device_destroy(perceptron_class, dev);
    class_destroy(perceptron_class);
    cdev_del(&perceptron_cdev);
    unregister_chrdev_region(dev, 1);
    pr_info("perceptron_kmod: unloaded\n");
}

module_init(perceptron_init);
module_exit(perceptron_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Tanisha");
MODULE_DESCRIPTION("Kernel-space fixed-point dot product accelerator for perceptron inference");