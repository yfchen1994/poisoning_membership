#!/bin/bash
#for encoder in inceptionv3 mobilenetv2 xception;
attack_type=dirty_label
device=3
for encoder in resnet50 vgg16 inceptionv3 mobilenetv2 xception;
do
    for (( i=0; i<10; i++ ));
    do
        for j in 400;
        do
            echo stl10 $i
            python poisoning_stl10.py --device_no $device --encoder $encoder --target_class $i --seed_amount $j --attack_type $attack_type 
        done
        
        for j in 1000;
        do
            echo mnist $i
            python poisoning_mnist.py --device_no $device --encoder $encoder --target_class $i --seed_amount $j --attack_type $attack_type 
            echo cifar10 $i
            python poisoning_cifar10.py --device_no $device --encoder $encoder --target_class $i --seed_amount $j --attack_type $attack_type
        done

        if (( i < 2 ));
        then
            for j in 5000;
            do
                echo patchcamelyon $i
                python poisoning_patchcamelyon.py --device_no $device --encoder $encoder --target_class $i --seed_amount $j --attack_type $attack_type
                echo celeba $i
                python poisoning_celeba.py --device_no $device --encoder $encoder --target_class $i --seed_amount $j --attack_type $attack_type
            done
        fi
    done
done