# model=clip
# CUDA_VISIBLE_DEVICES=0 python eval.py --model-type $model --mapper weights/mapper/${model}_tuned.pth --test-case test_observed_imagenet --result-path results/test_observed_imagenet/${model}_tuned.pkl &
# CUDA_VISIBLE_DEVICES=1 python eval.py --model-type $model --mapper weights/mapper/${model}_tuned.pth --test-case test_unobserved_imagenet --result-path results/test_unobserved_imagenet/${model}_tuned.pkl &
# CUDA_VISIBLE_DEVICES=2 python eval.py --model-type $model --mapper weights/mapper/${model}_tuned.pth --test-case test_bamfg --result-path results/test_bamfg/${model}_tuned.pkl &
# CUDA_VISIBLE_DEVICES=3 python eval.py --model-type $model --mapper weights/mapper/${model}_tuned.pth --test-case test_artchive --result-path results/test_artchive/${model}_tuned.pkl &
# model=dino
# CUDA_VISIBLE_DEVICES=4 python eval.py --model-type $model --mapper weights/mapper/${model}_tuned.pth --test-case test_observed_imagenet --result-path results/test_observed_imagenet/${model}_tuned.pkl &
# CUDA_VISIBLE_DEVICES=5 python eval.py --model-type $model --mapper weights/mapper/${model}_tuned.pth --test-case test_unobserved_imagenet --result-path results/test_unobserved_imagenet/${model}_tuned.pkl &
# CUDA_VISIBLE_DEVICES=6 python eval.py --model-type $model --mapper weights/mapper/${model}_tuned.pth --test-case test_bamfg --result-path results/test_bamfg/${model}_tuned.pkl &
# CUDA_VISIBLE_DEVICES=7 python eval.py --model-type $model --mapper weights/mapper/${model}_tuned.pth --test-case test_artchive --result-path results/test_artchive/${model}_tuned.pkl &
# wait $(jobs -p)

# model=moco
# CUDA_VISIBLE_DEVICES=0 python eval.py --model-type $model --mapper weights/mapper/${model}_tuned.pth --test-case test_observed_imagenet --result-path results/test_observed_imagenet/${model}_tuned.pkl &
# CUDA_VISIBLE_DEVICES=1 python eval.py --model-type $model --mapper weights/mapper/${model}_tuned.pth --test-case test_unobserved_imagenet --result-path results/test_unobserved_imagenet/${model}_tuned.pkl &
# CUDA_VISIBLE_DEVICES=2 python eval.py --model-type $model --mapper weights/mapper/${model}_tuned.pth --test-case test_bamfg --result-path results/test_bamfg/${model}_tuned.pkl &
# CUDA_VISIBLE_DEVICES=3 python eval.py --model-type $model --mapper weights/mapper/${model}_tuned.pth --test-case test_artchive --result-path results/test_artchive/${model}_tuned.pkl &
# model=vit
# CUDA_VISIBLE_DEVICES=4 python eval.py --model-type $model --mapper weights/mapper/${model}_tuned.pth --test-case test_observed_imagenet --result-path results/test_observed_imagenet/${model}_tuned.pkl &
# CUDA_VISIBLE_DEVICES=5 python eval.py --model-type $model --mapper weights/mapper/${model}_tuned.pth --test-case test_unobserved_imagenet --result-path results/test_unobserved_imagenet/${model}_tuned.pkl &
# CUDA_VISIBLE_DEVICES=6 python eval.py --model-type $model --mapper weights/mapper/${model}_tuned.pth --test-case test_bamfg --result-path results/test_bamfg/${model}_tuned.pkl &
# CUDA_VISIBLE_DEVICES=7 python eval.py --model-type $model --mapper weights/mapper/${model}_tuned.pth --test-case test_artchive --result-path results/test_artchive/${model}_tuned.pkl &
# wait $(jobs -p)

# model=sscd
# CUDA_VISIBLE_DEVICES=0 python eval.py --model-type $model --mapper weights/mapper/${model}_tuned.pth --test-case test_observed_imagenet --result-path results/test_observed_imagenet/${model}_tuned.pkl &
# CUDA_VISIBLE_DEVICES=1 python eval.py --model-type $model --mapper weights/mapper/${model}_tuned.pth --test-case test_unobserved_imagenet --result-path results/test_unobserved_imagenet/${model}_tuned.pkl &
# CUDA_VISIBLE_DEVICES=2 python eval.py --model-type $model --mapper weights/mapper/${model}_tuned.pth --test-case test_bamfg --result-path results/test_bamfg/${model}_tuned.pkl &
# CUDA_VISIBLE_DEVICES=3 python eval.py --model-type $model --mapper weights/mapper/${model}_tuned.pth --test-case test_artchive --result-path results/test_artchive/${model}_tuned.pkl &

model=clip
CUDA_VISIBLE_DEVICES=0 python eval.py --model-type $model --mapper weights/mapper/${model}_tuned.pth --test-case test_observed_imagenet --result-path results/test_observed_imagenet/${model}_tuned.pkl &
CUDA_VISIBLE_DEVICES=1 python eval.py --model-type $model --mapper weights/mapper/${model}_tuned.pth --test-case test_unobserved_imagenet --result-path results/test_unobserved_imagenet/${model}_tuned.pkl &

model=dino
CUDA_VISIBLE_DEVICES=2 python eval.py --model-type $model --mapper weights/mapper/${model}_tuned.pth --test-case test_observed_imagenet --result-path results/test_observed_imagenet/${model}_tuned.pkl &
CUDA_VISIBLE_DEVICES=3 python eval.py --model-type $model --mapper weights/mapper/${model}_tuned.pth --test-case test_unobserved_imagenet --result-path results/test_unobserved_imagenet/${model}_tuned.pkl &

model=vit
CUDA_VISIBLE_DEVICES=4 python eval.py --model-type $model --mapper weights/mapper/${model}_tuned.pth --test-case test_observed_imagenet --result-path results/test_observed_imagenet/${model}_tuned.pkl &
CUDA_VISIBLE_DEVICES=5 python eval.py --model-type $model --mapper weights/mapper/${model}_tuned.pth --test-case test_unobserved_imagenet --result-path results/test_unobserved_imagenet/${model}_tuned.pkl &

model=sscd
CUDA_VISIBLE_DEVICES=6 python eval.py --model-type $model --mapper weights/mapper/${model}_tuned.pth --test-case test_observed_imagenet --result-path results/test_observed_imagenet/${model}_tuned.pkl &
CUDA_VISIBLE_DEVICES=7 python eval.py --model-type $model --mapper weights/mapper/${model}_tuned.pth --test-case test_unobserved_imagenet --result-path results/test_unobserved_imagenet/${model}_tuned.pkl &
