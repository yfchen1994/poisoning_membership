#!/bin/bash
device = 2 
python poisoning_patchcamelyon.py --device_no $device --attack_type clean_label
python poisoning_celeba.py --device_no $device --attack_type clean_label
python poisoning_stl10.py --device_no $device --attack_type clean_label
python poisoning_mnist.py --device_no $device --attack_type clean_label
python poisoning_cifar10.py --device_no $device --attack_type clean_label