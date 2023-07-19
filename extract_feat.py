import os
import json
import pathlib
import argparse
import torch
from PIL import Image

import models


IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")
class FileDataset(torch.utils.data.Dataset):
    def __init__(self, path, transform, pattern='*', filelist=None):
        if filelist is None:
            path = pathlib.Path(path)
            files = [fname for fname in path.glob(pattern) if os.path.splitext(fname)[1] in IMG_EXTENSIONS]
            files = sorted(files)
        else:
            files = filelist

        self.num_images = len(files)
        self.files = files
        self.transform = transform

    def __getitem__(self, idx):
        path = self.files[idx]
        pil_image = Image.open(path).convert('RGB')
        return self.transform(pil_image)

    def __len__(self):
        return self.num_images


def collect_features(data_path, model, transform, batch_size=200, pattern='*', filelist=None, device='cuda'):
    dataset = FileDataset(data_path, transform, pattern=pattern, filelist=filelist)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=20
    )

    data_feat = []
    for ind, data in enumerate(data_loader):
        if ind % 100 == 0:
            print(f"Iter -- {ind}")
        with torch.no_grad():
            feat = model(data.to(device))
            data_feat.append(feat.cpu())
    data_feat = torch.cat(data_feat, dim=0).float()
    return data_feat, [str(n) for n in dataset.files]


def process_json_testcase(data_path, save_path, model, preprocess, batch_size, device='cuda'):
    """Special case for processing the test-case json file."""
    print(f'Reading json file {data_path}, extracting features from the test case...')

    json_name = data_path
    with open(json_name, 'r') as f:
        files = json.load(f)

    # Get all the synth types
    synth_types = list(files[0]['synth'].keys())

    all_data = {
        'exemplar_paths': [],                             # List of paths to exemplar images
        'exemplar_chunks': [],                            # List of tuples (start, end) indices, specifying exemplars used to train each Custom Diffusion model
        'exemplar_tensors': None,                         # Tensor of features for each exemplar image
        'synth_paths': {k: [] for k in synth_types},      # List of paths to synthesized images
        'synth_ids': {k: [] for k in synth_types},        # List of indices, specifying which Custom Diffusion model was used to synthesize each image
        'synth_tensors': {k: None for k in synth_types},  # Tensor of features for each synthesized image
    }

    # Collect all the paths and indices
    exemplar_len = 0
    for ind, d in enumerate(files):
        all_data['exemplar_paths'].extend(d['exemplar'])
        all_data['exemplar_chunks'].append((exemplar_len, exemplar_len + len(d['exemplar'])))
        exemplar_len += len(d['exemplar'])
        for k in synth_types:
            all_data['synth_paths'][k].extend(d['synth'][k])
            all_data['synth_ids'][k].extend([ind for _ in range(len(d['synth'][k]))])

    # Collect features for exemplars
    exemplar_feat = collect_features(None, model, preprocess, batch_size=batch_size, filelist=all_data['exemplar_paths'], device=device)[0]
    all_data['exemplar_tensors'] = exemplar_feat

    # Collect features for synthesized images
    for k in synth_types:
        synth_feat = collect_features(None, model, preprocess, batch_size=batch_size, filelist=all_data['synth_paths'][k], device=device)[0]
        all_data['synth_tensors'][k] = synth_feat

    # Save the data
    torch.save(all_data, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', type=str, required=True, help="Which base pretrained model to use.")
    parser.add_argument('--data-path', type=str, required=True, help="Folder to the images to extract features from. If --json is specified, this is the test-case json file instead.")
    parser.add_argument('--save-path', type=str, required=True, help='Where to save the extracted features.')
    parser.add_argument('--json', action='store_true', help='If specified, --data-path is a test-case json file instead of a folder.')
    parser.add_argument('--batch-size', type=int, default=200)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = args.device
    model_type = args.model_type
    batch_size = args.batch_size
    data_path = args.data_path
    save_path = args.save_path

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Load model and the image preprocessor that comes with it
    model = models.create_model(model_type).eval().to(device)
    preprocess = model.preprocess

    if args.json:
        # Extract features from the test case json file, and save into a .pth file.
        # We include features from exemplar images and Custom-Diffusion-synthesized images.
        process_json_testcase(data_path, save_path, model, preprocess, batch_size, device=device)
    else:
        # Extract features from the images in the folder
        print(f'Extracting features from {data_path}...')

        feats, paths = collect_features(data_path, model, preprocess, batch_size=batch_size, pattern='*', device=device)
        save_dict = {
            'feats': feats,
            'paths': paths,
        }
        torch.save(save_dict, save_path)
    print('Done!')
