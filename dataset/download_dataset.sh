#!/bin/bash
DATASET=$1

if [[ $DATASET == "testset" ]]; then
    gdown https://drive.google.com/uc?id=1z7wDJLvBGoW77ijh9HZQPZ9w7tfViCm- -O dataset/
    tar -xvf dataset/testset_synth.tar.gz -C dataset/
    rm dataset/testset_synth.tar.gz
    exit 0
fi

if [[ $DATASET == "exemplar" ]]; then
    gdown https://drive.google.com/uc?id=1UVYHxWM5D8Z7NWO9novYwHkwHKi7BsN0 -O dataset/
    tar -xvf dataset/exemplar.tar.gz -C dataset/
    rm dataset/exemplar.tar.gz
    exit 0
fi

if [[ $DATASET == "laion" ]]; then
    gdown https://drive.google.com/uc?id=1W4TEkxFwUwmNm6gMzgm7Wu6lb848CM-q -O dataset/
    tar -xvf dataset/laion_subset.tar.gz -C dataset/
    rm dataset/laion_subset.tar.gz
    exit 0
fi

if [[ $DATASET == "laion_jpeg" ]]; then
    gdown https://drive.google.com/uc?id=1n5HdyAomip1deqD7FTUN0GnoZA1QlwqX -O dataset/
    tar -xvf dataset/laion_subset_jpeg.tar -C dataset/
    rm dataset/laion_subset_jpeg.tar
    exit 0
fi