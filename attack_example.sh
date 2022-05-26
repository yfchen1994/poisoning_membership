# Launching clean-label poisoning attack
python poisoning_attack.py --target_class 0 --dataset cifar10 --encoder xception --seed_amount 1000 --attack_type clean_label
# Inspecting poisoning attack
python poisoning_attack.py --target_class 0 --dataset cifar10 --encoder xception --seed_amount 1000 --attack_type clean_label --check_mia true
# Launching dirty-label poisoning attack
python poisoning_attack.py --target_class 0 --dataset cifar10 --encoder xception --seed_amount 1000 --attack_type dirty_label
# Inspecting dirty-label poisoning attack
python poisoning_attack.py --target_class 0 --dataset cifar10 --encoder xception --seed_amount 1000 --attack_type dirty_label --check_mia true
