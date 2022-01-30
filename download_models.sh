#!/bin/bash
wget -O models/ldm/celeba256/celeba-256.zip https://ommer-lab.com/files/latent-diffusion/celeba.zip


cd models/ldm/celeba256
unzip -o celeba-256.zip

cd ../..

wget -O models/first_stage_models/vq-f4/model.zip https://ommer-lab.com/files/latent-diffusion/vq-f4.zip

cd models/first_stage_models/vq-f4
unzip -o model.zip

cd ../..
