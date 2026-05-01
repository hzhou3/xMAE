import pytorch_lightning as pl
import wandb
import argparse
import warnings
warnings.filterwarnings("ignore")

from utils.helper_dataloader import MyDataModule
from utils.helper_read_h5 import read_h5_from_s3
from utils.helper_general import read_from_cfg
from utils.helper_models import get_model
from utils.helper_trainer import get_trainer
from utils.helper_callbacks import get_callbacks

RANDOM = 'your_wandb_key'

def main(args):
    cfg = read_from_cfg(args)
    
    # --- Model Instantiation ---
    architecture = get_model(cfg)
    lightning_model = get_trainer(model=architecture, config=cfg)

    # --- Initialize W&B ---
    wandb.login(key=RANDOM)
    wandb.init(project="xMAE",
               config=cfg,
               name=cfg['experiment'])
    wandb_logger = pl.loggers.WandbLogger()
    
    # --- Data Loading ---
    dataset_in_memory = read_h5_from_s3(cfg)

    data_module = MyDataModule(h5_file=dataset_in_memory, cfg=cfg)
    
    # --- Configure Trainer with Callbacks ---
    trainer = pl.Trainer(
        precision="bf16-mixed",   
        max_epochs=cfg['train']['epochs'],
        accelerator='auto', devices='auto', strategy='auto',
        log_every_n_steps=20,
        logger=wandb_logger, 
        callbacks = get_callbacks(cfg)
    )
    
    # --- Run the Training ---
    print("\n-----> Training")
    trainer.fit(model=lightning_model, datamodule=data_module)
    print("<----- Finished.")

    # Finish the W&B run
    wandb.finish()


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        description="args parser in xMAE"
    )

    parser.add_argument(
        '-c',
        '--cfg', 
        type=str, 
        default='ppg-ecg', 
    )
    
    parser.add_argument(
        '-e',
        '--experiment',
        type = str, 
        default='test', 
    )
    
    args = parser.parse_args()

    args.cfg = f'./cfg/{args.cfg}.yaml'
    args.experiment = f'exp_{args.experiment}'
    
    main(args)



    