#include <linux/init.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/fs.h>
#include <linux/uaccess.h>

#include "exported_weights.h"

#define DEVICE_NAME "perceptron_kmod"
#define MAJOR_NUM 240

static int prediction = 0;

typedef struct
{
    int features[INPUT_SIZE];
} perceptron_input;

static long device_ioctl(struct file *file,
                         unsigned int cmd,
                         unsigned long arg)
{
    perceptron_input input;

    if(copy_from_user(&input,
                      (perceptron_input *)arg,
                      sizeof(input)))
        return -EFAULT;

    int activation = BIAS;

    for(int i=0;i<INPUT_SIZE;i++)
        activation += input.features[i] * WEIGHTS[i];

    prediction = (activation >= 0);

    return 0;
}

static ssize_t device_read(struct file *flip,
                           char __user *buffer,
                           size_t len,
                           loff_t *offset)
{
    if(copy_to_user(buffer,&prediction,sizeof(int)))
        return -EFAULT;

    return sizeof(int);
}

static struct file_operations fops={
    .owner=THIS_MODULE,
    .unlocked_ioctl=device_ioctl,
    .read=device_read
};

static int __init perceptron_init(void)
{
    register_chrdev(MAJOR_NUM,DEVICE_NAME,&fops);

    printk(KERN_INFO "Perceptron kernel loaded\n");

    return 0;
}

static void __exit perceptron_exit(void)
{
    unregister_chrdev(MAJOR_NUM,DEVICE_NAME);

    printk(KERN_INFO "Perceptron kernel removed\n");
}

module_init(perceptron_init);
module_exit(perceptron_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Tanisha Mathur");
MODULE_DESCRIPTION("Perceptron inference kernel using exported Python weights");