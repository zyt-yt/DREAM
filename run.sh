# cifar100 has 100-classes, tinyimagenet has 200-classes
# CUDA_VISIBLE_DEVICES=2 python3 condense.py --reproduce  -d cifar100 --ipc 10 --match grad


# when run in tinyimagenet, should change the normalization
# 1. argument.py: 442 change dir name 
# 2. condense.py: 309 remove the 255
# 3. condense.py: 574,666 set unnormalize = false
# CUDA_VISIBLE_DEVICES=3 python3 condense.py --reproduce  -d tinyimagenet -f 2 --ipc 10 --match grad --nclass 200
# CUDA_VISIBLE_DEVICES=0 python3 condense.py --reproduce  -d tinyimagenet --ipc 10 --match feat --nclass 200
# run in dream_tiny




# CUDA_VISIBLE_DEVICES=2 python3 condense.py --reproduce  -d cifar100 --ipc 10 --match feat 
# CUDA_VISIBLE_DEVICES=1 python3 condense.py --reproduce  -d tinyimagenet --ipc 10 --match feat --nclass 200

# DREAM+:
CUDA_VISIBLE_DEVICES=1 python condense_improve.py --reproduce  -d cifar100 -f 1 --ipc 10



