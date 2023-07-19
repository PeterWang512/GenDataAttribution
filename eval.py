import os
import argparse
import pickle
import torch
import numpy as np

import models
import networks


def average_precision(matches):
    """
    Computes the average precision, given the binary array of match indicators.
    """
    match_indices = np.where(matches)[0] + 1
    total_positive = len(match_indices)
    pos_count = np.arange(1, total_positive + 1)
    average_precision = np.sum(pos_count / match_indices) / total_positive
    return average_precision


def validate(real_mapper, synth_mapper, recall_ks, result_path, model_type, test_case, device, skip_mapping=False):
    print('loading')
    data_feat = torch.load(f'feats/{model_type}/laion_subset.pth', map_location=device)['feats']
    N = data_feat.shape[0]

    all_data = torch.load(f'feats/{model_type}/{test_case}.pth', map_location=device)
    exemplar_tensors = all_data['exemplar_tensors']
    synth_types = list(all_data['synth_tensors'].keys())

    # Map the features
    if not skip_mapping:
        data_feat = real_mapper(data_feat)
        exemplar_tensors = real_mapper(exemplar_tensors)
        for k in synth_types:
            all_data['synth_tensors'][k] = synth_mapper(all_data['synth_tensors'][k])

    # normalize features (for cosine similarity)
    data_feat = torch.nn.functional.normalize(data_feat, dim=-1, p=2)
    exemplar_tensors = torch.nn.functional.normalize(exemplar_tensors, dim=-1, p=2)
    for k in synth_types:
        all_data['synth_tensors'][k] = torch.nn.functional.normalize(all_data['synth_tensors'][k], dim=-1, p=2)

    with torch.no_grad():
        all_results = {}
        for k in synth_types:
            print(f"Processing {k}")
            synth_feat = all_data['synth_tensors'][k]
            idx_list = all_data['synth_ids'][k]
            sample_feat = synth_feat
            total_samples = len(sample_feat)

            ind = 0
            all_maps = []
            all_recalls = []
            for exemplar_id, sample_feat in zip(idx_list, sample_feat):
                if ind % 100 == 0:
                    print(f"Iter -- {ind} / {total_samples}")

                start, end = all_data['exemplar_chunks'][exemplar_id]
                exemplar_size = end - start

                add_exemplar = exemplar_tensors[start:end]
                all_feat = torch.cat([data_feat, add_exemplar], dim=0)
                score = all_feat @ sample_feat
                predictions = torch.argsort(-score).cpu()
                assert len(predictions.shape) == 1

                # compute average precision
                gt_matches = predictions >= N
                map_list = average_precision(gt_matches)

                # compute recall
                recall_list = []
                for max_pred in recall_ks:
                    recall_list.append((predictions[:max_pred] >= N).sum().item() / exemplar_size)

                all_maps.append(map_list)
                all_recalls.append(recall_list)

                del score
                del predictions
                ind += 1
            
            avg_map = np.array(all_maps).mean(axis=0)
            avg_recall = np.array(all_recalls).mean(axis=0)

            print(f"{model_type} {test_case} {k} MAP: {avg_map}")
            print(f"{model_type} {test_case} {k} Recall: {avg_recall}")

            all_results[k] = {
                'avg_map': avg_map,
                'avg_recall': avg_recall,
                'all_maps': all_maps,
                'all_recalls': all_recalls,
                'recall_ks': recall_ks,
            }

    with open(result_path, 'wb') as f:
        pickle.dump(all_results, f)
    
    return all_results


IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', type=str, required=True)
    parser.add_argument('--mapper-ckpt', type=str, default=None)
    parser.add_argument('--test-case', type=str, required=True)
    parser.add_argument('--result-path', type=str, required=True)
    parser.add_argument('--tune-type', type=str, default='linear')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--recall_ks', nargs='+', type=int, default=[5, 10, 100])

    # MLP args
    parser.add_argument('--mlp-hidden-dim', type=int, default=512)
    parser.add_argument('--mlp-out-dim', type=int, default=256)
    parser.add_argument('--mlp-layers', type=int, default=2)
    parser.add_argument('--mlp-dropout', type=float, default=0)
    args = parser.parse_args()

    device = args.device
    model_type = args.model_type
    mapper_ckpt = args.mapper_ckpt
    tune_type = args.tune_type
    test_case = args.test_case
    result_path = args.result_path
    recall_ks = args.recall_ks
    skip_mapping = mapper_ckpt is None
    torch.set_grad_enabled(False)

    os.makedirs(os.path.dirname(result_path), exist_ok=True)

    print(f'Testing on {test_case} ...')
    if skip_mapping:
        real_mapper, synth_mapper = None, None
    else:
        # Load the encoder
        model = models.create_model(model_type)
        if tune_type == 'mlp':
            mlp_kwargs = {
                'mlp_dim': args.mlp_hidden_dim,
                'output_dim': args.mlp_out_dim,
                'drop_out': args.mlp_dropout,
                'num_layers': args.mlp_layers,
            }
            real_encoder, synth_encoder = networks.create_encoders(model, tune_type, **mlp_kwargs)
        else:
            real_encoder, synth_encoder = networks.create_encoders(model, tune_type)

        real_mapper = real_encoder.mapper
        synth_mapper = synth_encoder.mapper
        real_mapper.eval().to(device)
        synth_mapper.eval().to(device)

        all_dicts = torch.load(mapper_ckpt)
        real_mapper.load_state_dict(all_dicts['real_mapper'])
        synth_mapper.load_state_dict(all_dicts['synth_mapper'])

    all_results = validate(real_mapper, synth_mapper, recall_ks, result_path, model_type, test_case, device, skip_mapping=skip_mapping)
    print(f"Results saved to {result_path}")
