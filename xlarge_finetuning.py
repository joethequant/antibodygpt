import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer
from models.progen.modeling_progen import ProGenForCausalLM
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import subprocess
from itertools import combinations
from seq import ab_number as abn
from sklearn.model_selection import KFold
import wandb
import argparse

wandb.login()

### configure fine-tuning
n_splits = 3
epochs = 10
batch_size = 20
learning_rate = 1e-5
foundation_model_name = 'progen2-xlarge' #progen2-small, progen2-medium, progen2-large, progen2-xlarge

fine_tuning_strategy = 'simple_fine_tuned' #None, simple_fine_tuned, frozen_layers_tuned, or etc
prompting_strategy = 'zero_shot' #zero_shot or prompted

if fine_tuning_strategy is None:
    model_name = f'{foundation_model_name}'
else:
    model_name = f'{fine_tuning_strategy}_{foundation_model_name}'
    
experiment_name =f'{model_name}_{prompting_strategy}'
run_description = f'Running {prompting_strategy} across {model_name}'

print(os.environ['LOCAL_RANK'])
# start a new wandb run to track this script
if int(os.environ['LOCAL_RANK']) == 0:
    wandb.init(
        # set the wandb project where this run will be logged
        project="berkeley_antibody_generation",
        entity='antibody_generation',
        name=experiment_name,
        notes=run_description,
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": learning_rate,
        "epochs": epochs,
        "foundational_model": foundation_model_name,
        "model_name": model_name,
        "run_description": run_description,
        "fine_tuning_strategy": fine_tuning_strategy
        }
    )


if fine_tuning_strategy == 'simple_fine_tuned':
    # Define a Dataset for loading protein sequences
    class ProteinDataset(Dataset):
        def __init__(self, sequences, tokenizer, begin_token_id, end_token_id):
            self.tokenized_sequences = [tokenizer.encode(sequence, add_special_tokens=False) for sequence in sequences]
            for i, encoding in enumerate(self.tokenized_sequences):
                modified_ids = [begin_token_id] + encoding.ids + [end_token_id]
                self.tokenized_sequences[i] = modified_ids
            
        def __len__(self):
            return len(self.tokenized_sequences)
    
        def __getitem__(self, idx):
            return self.tokenized_sequences[idx]
    
    def collate_fn(batch):
        # Find the max length of sequences in the batch
        max_length = max([len(sequence) for sequence in batch])
        
        # Pad each sequence to the max length and stack them
        padded_input_ids = torch.stack([torch.tensor(sequence + [0]*(max_length - len(sequence))) for sequence in batch])
        
        return {"input_ids": padded_input_ids, "labels": padded_input_ids.clone()}
    
    
    def main(epochs, batch_size, learning_rate, foundation_model_name):
        local_rank = int(os.environ['LOCAL_RANK'])
        dist.init_process_group(backend='nccl')

        torch.cuda.set_device(local_rank)
        # device = torch.device("cuda", local_rank)
        
        # Load the tokenizer
        tokenizer = Tokenizer.from_file('tokenizer.json')

        # Load the pre-trained model and move it to the GPU designated by local_rank
        model_path = f'./model_checkpoints/{foundation_model_name}'
        model = ProGenForCausalLM.from_pretrained(model_path)
        model.cuda(local_rank)

        # Wrap model with DistributedDataParallel
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
        
        # Load optimizer settings
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

        
        sequences = []
        # Open and read the file
        with open("sabdab_joint_sequences_uniprot.txt", "r") as file:
            [sequences.append(line.strip()) for line in file]

        # Create the KFold object
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # Splitting the data for train, and test.
        train_sequences, test_sequences = train_test_split(sequences, test_size=0.1, random_state=42)  # 70% train, 10% temp
        
        # Initialize test DataLoader for test set. Train and validation are handled in the training loop.
        test_dataset = ProteinDataset(test_sequences, tokenizer, begin_token_id=1, end_token_id=2)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
        for fold, (train_ids, val_ids) in enumerate(kf.split(sequences)):
            print(f"FOLD {fold}")
            print("--------------------------------")

            # Splitting the data into the current fold
            train_sequences = [sequences[index] for index in train_ids]
            val_sequences = [sequences[index] for index in val_ids]

            # Initialize DataLoaders for the current fold
            train_dataset = ProteinDataset(train_sequences, tokenizer, begin_token_id=1, end_token_id=2)
            val_dataset = ProteinDataset(val_sequences, tokenizer, begin_token_id=1, end_token_id=2)
    
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

            # Fine-tuning Loop
            model.train()
            for epoch in range(epochs):
                total_loss = 0.0
                for batch in train_loader:
                    optimizer.zero_grad()
                    inputs = batch["input_ids"].to(device)
                    labels = batch["labels"].to(device)
                    outputs = model(inputs, labels=labels)
                    loss = outputs.loss
        
                    #ig using multiple GPUs, This ensures that the loss is always a scalar, irrespective of the number of GPUs.
                    if loss.dim() > 0:
                        loss = loss.sum()
                    
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
        
                avg_loss = total_loss / len(train_loader)
                
                # Evaluate on validation set
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch in val_loader:
                        inputs = batch["input_ids"].to(device)
                        labels = batch["labels"].to(device)
                        outputs = model(inputs, labels=labels)
                        loss = outputs.loss
                        if loss.dim() > 0:
                            loss = loss.sum()
                        val_loss += loss.item()
                avg_val_loss = val_loss / len(val_loader)
                
                wandb.log({"avg_train_loss": avg_loss, "avg_val_loss": avg_val_loss})
                
                print(f"Epoch: {epoch+1}/{epochs}, Training Loss: {avg_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
                
                model.train()

            # Save the fine-tuned model
            if dist.get_rank() == 0:
                # Save only from the master process
                model_to_save = model.module if hasattr(model, 'module') else model
                model_to_save.save_pretrained(f'model_checkpoints/{foundation_model_name}_finetuned')


        # Evaluate on test set
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                inputs = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(inputs, labels=labels)
                loss = outputs.loss
                if loss.dim() > 0:
                    loss = loss.sum()
                test_loss += loss.item()
        avg_test_loss = test_loss / len(test_loader)
        wandb.log({"test_loss": avg_test_loss})
        print(f"Test Loss: {avg_test_loss:.4f}")

    if __name__ == '__main__':
        main(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, foundation_model_name=foundation_model_name)


