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
from config import get_weights_path,get_config,latest_weights_file_path

def greedy_decoder(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)



def get_all_sentences(ds,lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config,ds,lang):
    tokenizerpath = Path(config['tokenizer_file'].format(lang))
   
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
    tokenizers_src=get_or_build_tokenizer(config,ds_raw,config["src_lang"])
    target_tgt=get_or_build_tokenizer(config,ds_raw,config["tgt_lang"])
    train_ds_size=int(0.9*len(ds_raw))
    val_ds_size=len(ds_raw)-train_ds_size
    train_ds_raw,val_ds_raw=random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds=BilingualDataset(ds=train_ds_raw,tokenizer_src=tokenizers_src,tokenizer_tgt=target_tgt,src_lang=config["src_lang"],tgt_lang=config["tgt_lang"],seq_len=config["seq_len"])
    val_ds=BilingualDataset(ds=val_ds_raw,tokenizer_src=tokenizers_src,tokenizer_tgt=target_tgt,src_lang=config["src_lang"],tgt_lang=config["tgt_lang"],seq_len=config["seq_len"])
    
    max_len_src=max(len(tokenizers_src.encode(item['translation'][config['src_lang']]).ids) for item in ds_raw)
    max_len_tgt=max(len(target_tgt.encode(item['translation'][config['tgt_lang']]).ids) for item in ds_raw)
    print(f"max_len_src: {max_len_src}, max_len_tgt: {max_len_tgt}")
    
    train_dataloader=DataLoader(train_ds,batch_size=config["batch_size"],shuffle=True)
    val_dataloader=DataLoader(val_ds,batch_size=config["batch_size"],shuffle=False)
    return train_dataloader,val_dataloader,tokenizers_src,target_tgt
def get_model(config,vocab_src_len,vocab_tgt_len):
    model=build_transformer(src_vocab_size=vocab_src_len,tgt_vocab_size=vocab_tgt_len,src_seq_len=config["seq_len"],tgt_seq_len=config["seq_len"],d_model=config["d_model"],N=config["num_layers"],h=config["num_heads"],dropout=config["dropout"],d_ff=config["d_ff"])
    return model
def train_model(config):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("HAI CUDA??? O SEI UN COGLIONE IN CPU: ",torch.cuda.is_available())
    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)
    train_dataloader,val_dataloader,tokenizer_src,tokenizer_tgt=get_ds(config)
    model=get_model(config,vocab_src_len=tokenizer_src.get_vocab_size(),vocab_tgt_len=tokenizer_tgt.get_vocab_size()).to(device)
    writer=SummaryWriter(log_dir=f"runs/{config['experiment_name']}")
    optimizer=torch.optim.Adam(model.parameters(),lr=config["lr"],eps=1e-9)
    initial_epoch=0
    global_step=0
    preload=config["preload"]
    model_filename = latest_weights_file_path(config) if preload == 'latest' else latest_weights_file_path(config, preload) if preload else None

    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')
        
    loss_fn=nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id("[PAD]"),label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch,config["num_epochs"]):
        torch.cuda.empty_cache()
        model.train()
        total_loss=0
        batch_iterator=tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        if config["batch_size"]==1:
            run_validation(model,val_dataloader,tokenizer_src,tokenizer_tgt,config["seq_len"],device,lambda msg: batch_iterator.write(msg),global_state=global_step,writer=writer,num_samples=5)

        c=0
        for batch in batch_iterator: 
            encoder_input=batch["encoder_input"].to(device)
            decoder_input=batch["decoder_input"].to(device)
            encoder_mask=batch["encoder_mask"].to(device)
            decoder_mask=batch["decoder_mask"].to(device)
            labels=batch["label"].to(device)
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
            #################################
            #################################
            #################################
            if (global_step-5)%(1000//config["batch_size"]+1)==0:

                checkpoint_path=get_weights_path(config,f"epoch_{epoch+1}_step_{global_step}.pt")
                torch.save({
                    "epoch":epoch,
                    "global_step":global_step,
                    "model_state_dict":model.state_dict(),
                    "optimizer_state_dict":optimizer.state_dict(),
                },checkpoint_path)
                if config["batch_size"]==1:
                    run_validation(model,val_dataloader,tokenizer_src,tokenizer_tgt,config["seq_len"],device,lambda msg: batch_iterator.write(msg),global_state=global_step,writer=writer,num_samples=5)

        checkpoint_path=get_weights_path(config,f"epoch_{epoch+1}_step_{global_step}.pt")
        torch.save({
            "epoch":epoch,
            "global_step":global_step,
            "model_state_dict":model.state_dict(),
            "optimizer_state_dict":optimizer.state_dict(),
        },checkpoint_path)


        avg_loss=total_loss/len(train_dataloader)
        print(f"Epoch {epoch+1}/{config['num_epochs']}, Loss: {avg_loss:.4f}")
        writer.add_scalar("Loss/train",avg_loss,global_step)
  
def run_validation(model,validations_ds,tokenizer_src,tokenizer_tgt,max_len,device,print_msg,global_state,writer,num_samples=2):
    model.eval()
    count=0
    source_texts=[]
    expected=[]
    predicted=[]
    console_width=80
    with torch.no_grad():
        for batch in validations_ds:
            count+=1
            encoder_input=batch["encoder_input"].to(device)
            encoder_mask=batch["encoder_mask"].to(device)
            assert encoder_input.size(0)==1,"batch size must be 1"

            model_out=greedy_decoder(model,encoder_input,encoder_mask,tokenizer_src,tokenizer_tgt,max_len,device)
            source_text=batch["src_text"][0]
            target_text=batch["tgt_text"][0]
            model_out_text=tokenizer_tgt.decode(model_out.detach().cpu().numpy())
            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            print_msg("-"*console_width)
            print_msg(f"Source: {source_text}")
            print_msg(f"Expected: {target_text}")
            print_msg(f"Predicted: {model_out_text}")
            if count==num_samples:
                break

if __name__=="__main__":
    
    warnings.filterwarnings("ignore", category=UserWarning)
    config=get_config()
    train_model(config)