# KNO event reconstruction
KNO detector말고 다른 detector에서도 사용은 가능

## 전체적인 코드 실행

0. WCSim을 통해 MC 데이터 생성 (root 파일)

1. event_selection.py 코드를 이용하여 root파일을 h5 파일로 변환 (최초 한번만 진행)

2. training.py 코드를 이용하여 모델을 학습

    2.1 training.py 코드로 학습시 추가 학습이 필요하거나 학습이 중단됬을시 --transfer_learning 옵션을 이용하기

3. validation.py 코드를 이용하여 모델 평가를 위한 결과 뽑기

4. eval_*.ipynb 코드를 이용해서 성능확인인

## Conda install
    ### env_name에 원하는 콘다환경 이름 넣기
    conda create -n env_name python=3.11   
    
    ### pytorch.org에서 확인하고 본인 환경에 맞게 설치
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118   
    
    ### 최소한의 필요환경
    conda install conda-forge einops
    pip install h5py matplotlib jupyter ipykernel pandas uproot scikit-learn tqdm

    ### event selection 또는 root 파일로 뭔가 할때 필요요
    conda install -c conda-forge root

## Run example
    ###기본적인 vertex, direction, pid, energy 실행코드는 아래의 파일에 있음

    run_example.sh

## Training example
    python training.py --config config_elec.yaml -o vtx_electron_test \
                    --padding 1 --cla 3 --type 1 --device 0 --fea 5 \
                    --cross_dim 64 --self_head 8 --self_dim 64 --n_layers 5 \
                    --num_latents 200 --dropout_ratio 0.1 --nDataLoaders 8 --epoch 1 \
                    --batch 16 --learningRate 0.0001 --transfer_learning 0   

    python validation.py --config config_mu.yaml -o vtx_electron_test \
                    --padding 1 --cla 3 --type 1 --device 1 --nDataLoaders 8 --batch 80  

### training option 설명
- --config : config file의 이름을 넣으면 된다.
    - 예시로 config_all.yaml (pid에 사용) / config_elec.yaml / config_mu.yaml 이 있다.
    - 각 파일안에 본인이 사용할 데이터가 있는 path만 따로 설정해서 사용해주면 됨

- -o / -output : 아웃풋 폴더 이름 설정
- --type : pid, vertex, energy, direction 각 task를 결정
    -   pid 0 / vertex 1 / energy 2 / direction 3
    - exmaple :  vertex reconstruction을 하고 싶으면 아래처럼 설정하면 됨
        - --type 2
- --padding : charge가 없는 PMT들 정보를 어떻게 처리할지 정해줌
    - --padding 0 : charge가 0인 PMT들의 feature를 (0,0,PMT_position_x,PMT_position_y,PMT_position_z)로 넣게 된다.
    - --padding 1 : charge가 0인 PMT들의 feature를 (0,0,0,0,0)로 넣게 된다.
    - --padding을 1로 하여 PMT_position 정보를 없애야 PMT charge가 있는 정보들로만 학습을 하여 학습이 잘됨
- --cla : 모델의 마지막 output 크기를 지정
    - --cla 1 : pid와 energy시 사용
    - --cla 3 : vertex와 direction시 사용
- --device : 사용을 gpu 번호를 설정해줌 (각 컴퓨팅환경에 맞추어설정)
- --multi_device : multi gpu 사용시에만 사용. 단일 gpu 사용시 설정하면 됨
    - multi_gpu 사용시의 코드는 training.py 코드에 def main_multi_gpu이란 이름으로 만들어져있지만 주석처리되어 있음
    - 사용을 원할지 main_one_gpu코드에 맞추어 변경하게 사용하면 됨됨
- --transfer_learning : 학습이 끝난 모델에서 추가로 더 학습을 하고 싶을때 사용
    - --transfer_learning 0 : 0으로 설정하는 최초 실행시에만 사용
    - --transfer_learning N : N이 100일 경우, 100epoch 까지 학습할 모델을 100에서 부터 추가 학습시 이용한다.
- --rank_i : multi_gpu 학습시 필요. 그냥 내버려 두면 된다.
- --fea : 모델에 들어각 각 PMT의 feature수
    - (charge, time, PMT_position_x,PMT_position_y,PMT_position_z)로 5가 기본 설정
- --cross_head : cross_attention block의 head수
    - --cross_head 1 : 1이외의 세팅에서는 잘 학습이 되지 않음 1로 하는걸 추천
- --cross_dim : cross_attention block의 feature(차원)수
    - 32 / 64 / 128을 주로 사용함
- --self_head : self_attention block의 head수
    - 4 / 8 / 16을 주로 사용함
- --self_dim : self_attention block의 feature(차원)수
    - 64 / 128 / 256을 주로 사용함
- --n_layers : self_attention block의 수
    - 5 / 7 / 9를 주로 사용함
- --num_latents : cross_attention의 latent array 크기를 결정
    - 200 / 500 을 주로 사용
    - KNO기준 30912개의 PMT를 200(500) 개로 압축한다고 생각하면 됨
- --dropout_ratio : Fully connected layer에서의 drouput 비율 설정
- --nDataLoaders : 데이터 전처리에 사용될 cpu core수. 4 / 8 중 본인의 컴퓨터 환경에 따라 설정
- --epoch : 모델이 전체데이터에 대하여 몇번의 학습할지 설정.
- --batch : 학습시 한번에 학습할 event수. 본인의 gpu환경에 맞춰 설정하면 됨
- --learningRate : 학습률. 0.0001을 기본으로 설정해두고 사용하면 된다.
- --randomseed : 데이터를 random 하게 섞기기
    
## Loss check
기본적으로 두 코드 모두 path 설정은 따로 해주어야 한다.
- loss_pid.ipynb : pid loss만 확인(accuracy도 같이)
- loss_other.ipynb : pid를 제외한 task들의 loss확인

## Evaluation
eval_*.ipynb 코드는 모두 각 task들의 결과를 확인하는 코드다.
최소한의 결과만 보도록 되어 있어 본인이 맞게 수정을 하면됨.


## Event Selection
WCSim에서 만듬 root파일을 모델 학습을 위해 h5파일로 변환한다.
그 과정에서 event selection도 같이 진행

    python event_selection.py -i root_file.root -o h5_file.h5

event_selection.py 파일 안데 libWCSimRoot.so파일 경로 설정은 각각의 WCSim 환경에 맞춰 설정
 
 ROOT.gSystem.Load("path/libWCSimRoot.so")