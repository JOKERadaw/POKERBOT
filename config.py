from pathlib import Path
def get_config():
    config={
        "src_lang":"en",
        "tgt_lang":"it",
        "seq_len":350,
        "d_model":512,
        "num_layers":6,
        "num_heads":8,
        "dropout":0.1,
        "d_ff":2048,
        "batch_size":1,
        "model_folder":"model",
        "model_file_name":"ADAW",
        "src_tokenizer_path":"tokenizer_src.json",
        "tgt_tokenizer_path":"tokenizer_tgt.json",
        "lr":1e-4,
        "num_epochs":10000,
        "preload":None,
        "tokenizer_file":"tokenizer_{0}.json",
        "experiment_name":"transformer_experiment",
        "datasource": 'model',
        "preload": "latest"
    }
    return config
def get_weights_path(config, epoch: str):
    model_folder = f"{config['model_folder']}"
    model_filename = f"{config['model_file_name']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)
def latest_weights_file_path(config):
    model_folder = f"{config['model_folder']}"
    model_filename = f"{config['model_file_name']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])