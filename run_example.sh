#!/bin/bash


#### vertex electron example
python training.py --config config_elec.yaml -o vtx_electron_test \
                    --padding 1 --cla 3 --type 1 --device 0 --fea 5 \
                    --cross_dim 64 --self_head 8 --self_dim 64 --n_layers 5 \
                    --num_latents 200 --dropout_ratio 0.1 --nDataLoaders 8 --epoch 1 \
                    --batch 16 --learningRate 0.0001 --transfer_learning 0            

#### vertex muon example
python training.py --config config_mu.yaml -o vtx_mu_test \
                    --padding 1 --cla 3 --type 1 --device 1 --fea 5 \
                    --cross_dim 64 --self_head 8 --self_dim 64 --n_layers 5 \
                    --num_latents 200 --dropout_ratio 0.1 --nDataLoaders 8 --epoch 2 \
                    --batch 16 --learningRate 0.0001 --transfer_learning 0  

python validation.py --config config_mu.yaml -o vtx_mu_test \
                    --padding 1 --cla 3 --type 1 --device 1 --nDataLoaders 8 --batch 80  


#### pid example
python training.py --config config_all.yaml -o pid_test \
                    --padding 1 --cla 1 --type 0 --device 1 --fea 5 \
                    --cross_dim 64 --self_head 8 --self_dim 64 --n_layers 5 \
                    --num_latents 200 --dropout_ratio 0.1 --nDataLoaders 8 --epoch 2 \
                    --batch 16 --learningRate 0.0001 --transfer_learning 0                                                    

python validation.py --config config_mu.yaml -o pid_test \
                    --padding 1 --cla 1 --type 0 --device 1 --nDataLoaders 8 --batch 80  

#### direction muon example
python training.py --config config_mu.yaml -o direction_mu_test \
                    --padding 1 --cla 3 --type 3 --device 0 --fea 5 \
                    --cross_dim 64 --self_head 8 --self_dim 64 --n_layers 5 \
                    --num_latents 200 --dropout_ratio 0.1 --nDataLoaders 8 --epoch 2 \
                    --batch 16 --learningRate 0.0001 --transfer_learning 0  

python validation.py --config config_mu.yaml -o direction_mu_test \
                    --padding 1 --cla 3 --type 3 --device 1 --nDataLoaders 8 --batch 80  

#### energy muon example
python training.py --config config_mu.yaml -o energy_mu_test \
                    --padding 1 --cla 1 --type 2 --device 0 --fea 5 \
                    --cross_dim 64 --self_head 8 --self_dim 64 --n_layers 5 \
                    --num_latents 200 --dropout_ratio 0.1 --nDataLoaders 8 --epoch 2 \
                    --batch 16 --learningRate 0.0001 --transfer_learning 0  

python validation.py --config config_mu.yaml -o energy_mu_test \
                    --padding 1 --cla 1 --type 2 --device 1 --nDataLoaders 8 --batch 80                      