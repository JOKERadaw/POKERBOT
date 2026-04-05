import warnings

import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset,random_split
from tqdm import tqdm
from datasets import load_dataset
from dataset import BilingualDataset,causal_mask

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from model import build_transformer
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter
from model import build_transformer
from config import get_weights_path,get_config



def get_all_sentences(ds,lang):
    for item in ds:
        yield item[f'{lang}_text']

def get_or_build_tokenizer(config,ds,lang):
    tokenizerpath=Path(config[f'{lang}_tokenizer_path'])
    if not Path.exists(tokenizerpath):
        tokenizer=Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer=Whitespace()
        trainer=WordLevelTrainer(special_tokens=["[UNK]","[PAD]","[SOS]","[EOS]"],min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds,lang),trainer=trainer)
        tokenizer.save(str(tokenizerpath))
    else:
        tokenizer=Tokenizer.from_file(str(tokenizerpath))
    return tokenizer
def get_ds(config):
    ds_raw=load_dataset("opus_books",f"{config['src_lang']}-{config['tgt_lang']}",split="train" )
    tokenizers_src=get_or_build_tokenizer(config,ds_raw,config["lang_src"])
    target_tgt=get_or_build_tokenizer(config,ds_raw,config["lang_tgt"])
    train_ds_size=int(0.9*len(ds_raw))
    val_ds_size=len(ds_raw)-train_ds_size
    train_ds_raw,val_ds_raw=ds_raw.train_test_split(train_size=train_ds_size,test_size=val_ds_size)
    train_ds=BilingualDataset(tokenizer_src=tokenizers_src,tokenizer_tgt=target_tgt,src_lang=config["lang_src"],tgt_lang=config["lang_tgt"],seq_len=config["seq_len"])
    val_ds=BilingualDataset(tokenizer_src=tokenizers_src,tokenizer_tgt=target_tgt,src_lang=config["lang_src"],tgt_lang=config["lang_tgt"],seq_len=config["seq_len"])
    max_len_src=max(len(tokenizers_src.encode(item[f'{config["lang_src"]}_text']).ids) for item in ds_raw)
    max_len_tgt=max(len(target_tgt.encode(item[f'{config["lang_tgt"]}_text']).ids) for item in ds_raw)
    print(f"max_len_src: {max_len_src}, max_len_tgt: {max_len_tgt}")
    
    train_dataloader=DataLoader(train_ds,batch_size=config["batch_size"],shuffle=True)
    val_dataloader=DataLoader(val_ds,batch_size=config["batch_size"],shuffle=False)
    return train_dataloader,val_dataloader,tokenizers_src,target_tgt
def get_model(config,vocab_src_len,vocab_tgt_len):
    model=build_transformer(src_vocab_size=vocab_src_len,tgt_vocab_size=vocab_tgt_len,src_seq_len=config["seq_len"],tgt_seq_len=config["seq_len"],d_model=config["d_model"],N=config["num_layers"],h=config["num_heads"],dropout=config["dropout"],d_ff=config["d_ff"])
    return model
def train_model(config):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)
    train_dataloader,val_dataloader,tokenizer_src,tokenizer_tgt=get_ds(config)
    model=get_model(config,vocab_src_len=tokenizer_src.get_vocab_size(),vocab_tgt_len=tokenizer_tgt.get_vocab_size())
    writer=SummaryWriter(log_dir=f"runs/{config['experiment_name']}")
    optimizer=torch.optim.Adam(model.parameters(),lr=config["lr"],eps=1e-9)
    initial_epoch=0
    global_step=0
    if config["preload"] is not None:
        checkpoint=torch.load(config["preload"])
        print(f"Resuming training from epoch {checkpoint['epoch']} and global step {checkpoint['global_step']}")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        initial_epoch=checkpoint["epoch"]+1
        global_step=checkpoint["global_step"]
        
    loss_fn=nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id("[PAD]"),label_smoothing=0.1).to(device)
    for epoch in range(initial_epoch,config["num_epochs"]):
        model.train()
        total_loss=0
        batch_iterator=tqdm(train_dataloader,desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        for batch in batch_iterator:
            encoder_input=batch["encoder_input"].to(device)
            decoder_input=batch["decoder_input"].to(device)
            encoder_mask=batch["encoder_mask"].to(device)
            decoder_mask=batch["decoder_mask"].to(device)
            labels=batch["labels"].to(device)
            optimizer.zero_grad()
            encoder_output=model.encode(encoder_input,encoder_mask)
            decoder_output=model.decode(encoder_output,encoder_mask,decoder_input,decoder_mask)
            projected_output=model.project(decoder_output)
            loss=loss_fn(projected_output.view(-1,projected_output.size(-1)),labels.view(-1))
            batch_iterator.set_postfix({"loss": loss.item()})
            writer.add_scalar("Loss/batch",loss.item(),global_step)
            writer.flush()
            loss.backward()
            optimizer.step()
            total_loss+=loss.item()
            global_step+=1
        avg_loss=total_loss/len(train_dataloader)
        print(f"Epoch {epoch+1}/{config['num_epochs']}, Loss: {avg_loss:.4f}")
        writer.add_scalar("Loss/train",avg_loss,global_step)
        checkpoint_path=get_weights_path(config,f"epoch_{epoch+1}_step_{global_step}.pt")
        torch.save({
            "epoch":epoch,
            "global_step":global_step,
            "model_state_dict":model.state_dict(),
            "optimizer_state_dict":optimizer.state_dict(),
        },checkpoint_path)

if __name__=="__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    config=get_config()
    train_model(config)