import os
import torch
import numpy as np
import streamlit as st
from PIL import Image

import models
import networks


torch.set_grad_enabled(False)
st.set_page_config(layout="wide")
st.session_state.old_buffer = None


@st.cache(hash_funcs={torch.nn.parameter.Parameter: lambda _: None, torch.Tensor: lambda _: None}, allow_output_mutation=True)
def load_model(model_type, mapper_ckpt):
    model = models.create_model(model_type).eval().to(device)
    real_encoder, synth_encoder = networks.create_encoders(model, 'linear')

    real_mapper = real_encoder.mapper
    synth_mapper = synth_encoder.mapper
    real_mapper.eval().to(device)
    synth_mapper.eval().to(device)

    all_dicts = torch.load(mapper_ckpt, map_location=device)
    real_mapper.load_state_dict(all_dicts['real_mapper'])
    synth_mapper.load_state_dict(all_dicts['synth_mapper'])
    return model, real_mapper, synth_mapper


@st.cache(hash_funcs={torch.nn.parameter.Parameter: lambda _: None, torch.Tensor: lambda _: None})
def load_laion_subset(model_type, real_mapper, device):
    laion_dict = torch.load(
        f'feats/{model_type}/laion_subset.pth', map_location=device)
    laion_paths = [n.replace('png', 'jpg') for n in laion_dict['paths']]
    data_feat = laion_dict['feats']
    data_feat_lin = real_mapper(data_feat)
    data_feat_lin = normalize_feat(data_feat_lin)
    return data_feat_lin, laion_paths


def normalize_feat(feat):
    return torch.nn.functional.normalize(feat, dim=-1, p=2)


def load_img(buffer):
    image = np.array(Image.open(buffer).convert("RGB"))
    return image


def resize_img(img, scale=None, shortside=None, resample=Image.BICUBIC):
    # if scale is specified, use it
    # otherwise, resize to short side
    img_pil = Image.fromarray(img)
    H, W = img.shape[0], img.shape[1]
    assert(scale is not None or shortside is not None)
    if scale is None:  # use size
        minHW = min(H, W)
        scale = 1. * shortside / minHW

    Hnew, Wnew = int(H * scale), int(W * scale)
    img_pil = img_pil.resize((Wnew, Hnew), resample=resample)
    return np.array(img_pil)


# Settings
with st.sidebar:
    st.title('Attribution demo')

    device = 'cuda'
    laion_folder = 'dataset/laion_subset_jpeg'

    st.header('Model')
    method_types = ['clip (finetuned)', 'dino (finetuned)']
    method_type = st.select_slider(
        'Attribution features', options=method_types, value='clip (finetuned)')
    model_type = method_type.split(' ')[0]

    img_file_buffer = st.file_uploader(
        "Upload an image", type=["png", "jpg", "jpeg", 'webp'])
    if img_file_buffer is not None and img_file_buffer != st.session_state.old_buffer:
        st.session_state.old_buffer = img_file_buffer
        img = load_img(img_file_buffer)
        st.session_state.img = img
    else:
        img = load_img('images/test.png')

    st.header('Display')
    N = st.select_slider('Number retrieved', options=(
        10**np.linspace(1, 3, 101)).astype('int'), value=100)
    num_columns = st.slider('Width', min_value=1, max_value=20, value=5)

    res = st.select_slider(
        'Resolution', options=np.arange(64, 1024+32, 32), value=256)
    pad = st.slider('Padding', min_value=0, max_value=64, value=8)
    display_per = 0.007

    mapper_ckpt = f'weights/mapper/{model_type}_tuned.pth'
    calib_ckpt = f'weights/calibrator/{model_type}_tuned.pth'
    calibrator = networks.Calibrator()
    calibrator.load_state_dict(torch.load(calib_ckpt))

    st.write(
        model_type, f'inv temp: {calibrator.inv_tau.item():.4f}; bias: {calibrator.get_bias().item():.4f}')


# base model
model, real_mapper, synth_mapper = load_model(model_type, mapper_ckpt)

# load training image features and paths
data_feat_lin, laion_paths = load_laion_subset(model_type, real_mapper, device)

img_disp = resize_img(img, shortside=res)
st.image(resize_img(img, shortside=res), width=img_disp.shape[1])

img_query = model.preprocess(Image.fromarray(img)).unsqueeze(0).to(device)
query = model(img_query).float()

query_feat_lin = synth_mapper(query)
query_feat_lin = normalize_feat(query_feat_lin)

# Features --> Scores
# N x F (N = number of training images, F = features, 512 for CLIP)
score = data_feat_lin @ query_feat_lin.squeeze(0)  # N, similarity scores

# sort & softmax
predictions = torch.argsort(-score).cpu()  # sort by score
sorted_score = score[predictions]  # highest scores
calibrated_softmax = calibrator(sorted_score.unsqueeze(0))[0]

new_row = True
prev_ind = 0
for i in range(N):
    filename = laion_paths[predictions[i]]
    img = load_img(os.path.join(laion_folder, filename))
    shortside = 1 + \
        int((res - 1) * torch.sqrt(calibrated_softmax[i] / display_per))
    img_rs = resize_img(img, shortside=shortside)

    if new_row:
        img_accum = img_rs
        new_row = False
    else:
        H_accum, W_accum = img_accum.shape[0], img_accum.shape[1]
        H_new, W_new = img_rs.shape[0], img_rs.shape[1]
        img_accum = np.concatenate(
            [img_accum, 255 + np.zeros((H_accum, W_new + pad, 3), dtype=img_accum.dtype)], axis=1)
        img_accum[int((H_accum - H_new) / 2):int((H_accum - H_new) / 2) +
                  H_new, W_accum + pad:, :] = img_rs

        if img_accum.shape[1] > num_columns * res:
            new_row = True
            next_ind = i + 1
            aggregated_score = torch.sum(calibrated_softmax[prev_ind : next_ind]).item()
            st.write(
                f'**[{aggregated_score * 100:.2f}%] estimated influence**, **{next_ind - prev_ind} images**')
            prev_ind = next_ind
            st.image(img_accum, width=img_accum.shape[1])

remaining = 1 - torch.sum(calibrated_softmax[:N]).item()
st.write('Remaining influence:', f'{remaining * 100:.2f}%')
