# xMAE: Physiology-Aware Masked Cross-Modal Reconstruction for Biosignal Representation Learning (ICML'26)

## Repo Structure
```
.
├── cfg
│   └── xmae.yaml
├── preprocessing
│   └── pulsedb.py
├── utils
│   ├── helper_callbacks.py
│   ├── helper_dataloader.py
│   ├── helper_general.py
│   ├── helper_logger.py
│   ├── helper_models.py
│   ├── helper_read_h5.py
│   └── helper_trainer.py
├── model_arch
│   └── xmae.py
├── Dockerfile
├── eval_0_simple_example.ipynb
├── eval_1_pvc.ipynb
├── pretrain.py
├── pvc_10s_synth.h5
├── pvc_10s_synth_metadata.csv
├── README.md
└── xmae_weights_permute.pth
```

## Installation
xMAE is built with Python 3.10+ with NVIDIA H200 GPUs; Please follow `Dockerfile` to replicate the enviroment.

## Downloading Pretraining Data
0. Follow [here](https://physionet.org/content/mimic3wdb-matched/1.0/) or [here](https://github.com/pulselabteam/PulseDB) to download the dataset.

1. We provide the script we used for processing the downloaded dataset in `preprocesing\process.py`. You need to update the variables `S3_BUCKET` and `DOWNLOADED_DATA` in the python file. This script includes our full signal preprocessing steps.


## Pretraining
`python3 pretrain.py -c xmae -e this-is-a-test > output.log`


## Runnables
0. We provide a minimal example to build and load xMAE and check its size, etc in `eval_0_simple_example.ipynb`.
1. We provide synthetic PVCs and a notebook to load and linear probe. See `eval_1_pvc.ipynb`.


## Notes
0. The weights and data provided in this repo are not real. We are unable to release weights and data due to industrial policy. During the span of ICML review, we will keep updating the repo and try to release as much as we could under the industrial policy. 
1. The preprocessing code, and pretrain code should allow interested parties to reproduce xMAE.
2. `*.ipynb` can be seen for quick evaluation pipeline.



