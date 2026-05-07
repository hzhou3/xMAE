# xMAE (ICML'26)

Official implementation of the paper: xMAE "Physiology-Aware Masked Cross-Modal Reconstruction for Biosignal Representation Learning"(https://arxiv.org/abs/2605.00973)


![Overview of xMAE. (1) Pretraining: the model learns physiological structure by progressively reconstructing continuously masked ECG segments from synchronized PPG via directional cross-attention, encouraging the PPG encoder to capture underlying cardiac dynamics. (2) Evaluation: the PPG encoder is transferred to downstream tasks spanning cardiovascular conditions, sleep staging, blood lab results, and demographics across 6 studies (19 tasks; 2.3k hours of PPG; 12.5k subjects). (3) Performance: Despite a smaller pretraining data scale, xMAE achieves higher averaged classification performance compared to prior open-source foundation models.](misc/arch.png)



---

### 🛠️ Building from scratch


##### 0. Repo Structure
```
.
├── cfg
│   └── xmae.yaml
├── preprocessing
│   └── process.py 
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
#### 1. Environment Setup
xMAE is built with Python 3.10+ with NVIDIA H200 GPUs; Please follow `Dockerfile` to replicate the enviroment.

#### 2. Downloading Pretraining Data
0. Follow [here](https://physionet.org/content/mimic3wdb-matched/1.0/) or [here](https://github.com/pulselabteam/PulseDB) to download the dataset.

1. We provide the script we used for processing the downloaded dataset in `preprocesing\process.py`. You need to update the variables `S3_BUCKET` and `DOWNLOADED_DATA` in the python file. This script includes our full signal preprocessing steps.


#### 3. Pretraining
`python pretrain.py -c xmae -e experiment-name > output.log`

`-c`: reading cfg from `cfg/xmae.yaml`

`-e`: saving weights to the folder named `experiment-name`

#### 4. Runnables
0. `eval_0_simple_example.ipynb`: a minimal example to build, load xMAE and check its size, etc.
1. `eval_1_pvc.ipynb`: a notebook to load and linear probe synthetic PVCs.


#### 5. Notes
0. We are unable to release weights and data due to industrial policy. Thus, `h5` and `pth` files are made-up.
1. The preprocessing code, and pretrain code should allow interested parties to reproduce xMAE.
2. `*.ipynb` can be seen for quick evaluation pipeline.



---

## 📖 Cite

If you find this repo or our paper useful, please cite our work
```
@misc{xmae,
      title={Physiology-Aware Masked Cross-Modal Reconstruction for Biosignal Representation Learning}, 
      author={Hao Zhou and Simon A. Lee and Cyrus Tanade and Keum San Chun and Juhyeon Lee and Migyeong Gwak and Megha Thukral and Justin Sung and Eugene Hwang and Mehrab Bin Morshed and Li Zhu and Viswam Nathan and Md Mahbubur Rahman and Subramaniam Venkatraman and Sharanya Arcot Desai},
      year={2026},
      eprint={2605.00973},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.00973}, 
}
```


