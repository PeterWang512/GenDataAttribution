for model in dino clip moco vit sscd
do
    python eval.py --model-type $model --mapper weights/mapper/${model}_tuned.pth --test-case test_observed_imagenet --result-path results/test_observed_imagenet/${model}_tuned.pkl
    python eval.py --model-type $model --mapper weights/mapper/${model}_tuned.pth --test-case test_unobserved_imagenet --result-path results/test_unobserved_imagenet/${model}_tuned.pkl
    python eval.py --model-type $model --mapper weights/mapper/${model}_tuned.pth --test-case test_bamfg --result-path results/test_bamfg/${model}_tuned.pkl
    python eval.py --model-type $model --mapper weights/mapper/${model}_tuned.pth --test-case test_artchive --result-path results/test_artchive/${model}_tuned.pkl
done