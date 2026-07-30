# Build both kernel modules

obj-m += perceptron_kernel.o
obj-m += perceptron_kmod.o

KDIR := /lib/modules/$(shell uname -r)/build
PWD := $(shell pwd)

all:
	make -C $(KDIR) M=$(PWD) modules

clean:
	make -C $(KDIR) M=$(PWD) clean