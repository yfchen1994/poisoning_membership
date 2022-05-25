#!/bin/bash
echo "Mount poisoning attacks"

for attack in dirty_label clean_label;
do
    echo "Attack: $attack"
    for encoder in xception resnet50 inceptionv3 mobilenetv2 vgg16;
    do
        for dataset in mnist cifar10;
        do
            echo "Dataset $dataset"
            for (( i=0; i<10; i++ ));
            do
                echo "Target class: $i"
                python poisoning_attack.py --target_class $i --dataset $dataset --encoder $encoder --seed_amount 1000 --attack_type $attack --check_mia True
            done
        done

        for dataset in stl10;
        do
            echo "Dataset $dataset"
            for (( i=0; i<10; i++ ));
            do
                echo "Target class: $i"
                python poisoning_attack.py --target_class $i --dataset $dataset --encoder $encoder --seed_amount 400 --attack_type $attack --check_mia True 
            done
        done

        for dataset in celeba patchcamelyon;
        do
            echo "Dataset $dataset"
            for (( i=0; i<1; i++ ));
            do
                echo "Target class: $i"
                python poisoning_attack.py --target_class $i --dataset $dataset --encoder $encoder --seed_amount 5000 --attack_type $attack --check_mia True
            done
        done
    done
done