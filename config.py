from pathlib import Path
def get_config():
    config={
        "src_lang":"en",
        "tgt_lang":"it",
        "seq_len":128,
        "d_model":512,
        "num_layers":6,
        "num_heads":8,
        "dropout":0.1,
        "d_ff":2048,
        "batch_size":64,
        "model_folder":"model",
        "model_file_name":"ADAW",
        "src_tokenizer_path":"tokenizer_src.json",
        "tgt_tokenizer_path":"tokenizer_tgt.json",
        "lr":1e-4,
        "num_epochs":10,
        "preload":None,
        "tokenizer_file":"tokenizer.json",
        "experiment_name":"transformer_experiment"
    }
    return config
def get_weights_path(config):
    return f"{config['model_folder']}/{config['model_file_name']}.pth"