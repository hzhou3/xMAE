def get_model(cfg: dict):
        
    if cfg['model'] == 'xmae'.lower():
        from utils.model_arch.xmae import build_model_from_cfg
        return build_model_from_cfg(cfg)        

    else:
        print('No such type of model...')
        exit()







        
        