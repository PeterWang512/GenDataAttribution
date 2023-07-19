for model in dino clip moco vit sscd
do
    python extract_feat.py --model-type $model --json --data-path dataset/json/test_observed_imagenet.json --save-path feats/$model/test_observed_imagenet.pth
    python extract_feat.py --model-type $model --json --data-path dataset/json/test_unobserved_imagenet.json --save-path feats/$model/test_unobserved_imagenet.pth
    python extract_feat.py --model-type $model --json --data-path dataset/json/test_bamfg.json --save-path feats/$model/test_bamfg.pth
    python extract_feat.py --model-type $model --json --data-path dataset/json/test_artchive.json --save-path feats/$model/test_artchive.pth
done
