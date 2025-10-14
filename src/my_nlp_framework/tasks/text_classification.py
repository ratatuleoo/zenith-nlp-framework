import torch
import torch.nn as nn
from torch.utils.data import random_split
import pandas as pd
from functools import partial
import hydra
from omegaconf import DictConfig

from ..data.loaders import GenericNLPDataset
from ..core.bpe_tokenizer import BPETokenizer
from ..models.bert_model import BERT
from ..training.training import train_model

class TextClassifier(nn.Module):
    def __init__(self, backbone, embed_size, num_classes):
        super(TextClassifier, self).__init__()
        self.backbone = backbone
        self.fc = nn.Linear(embed_size, num_classes)
        
    def forward(self, ids, segment_info=None):
        if isinstance(self.backbone, BERT):
            features = self.backbone(ids, segment_info)[:, 0, :]
        else:
            features = self.backbone(ids)[:, -1, :]
        return self.fc(features)

def process_classification_data(row, tokenizer, max_len):
    """Processor function for text classification."""
    text = str(row.text)
    label = row.label
    
    token_ids = tokenizer.tokenize(text)
    
    if len(token_ids) < max_len:
        token_ids += [0] * (max_len - len(token_ids))
    else:
        token_ids = token_ids[:max_len]
        
    return {
        "ids": torch.tensor(token_ids, dtype=torch.long),
        "segment_info": torch.ones(max_len, dtype=torch.long),
        "label": torch.tensor(label, dtype=torch.long)
    }

@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig):
    df = pd.read_csv(cfg.data_path).dropna()
    
    if len(df) < cfg.sample_size:
        df_sample = df.copy()
    else:
        df_sample = df.sample(n=cfg.sample_size, random_state=42)

    df_sample['label'] = torch.randint(0, cfg.num_classes, (df_sample.shape[0],)).tolist()

    print("Training BPE tokenizer...")
    tokenizer = BPETokenizer()
    tokenizer.train(df_sample['text'].tolist(), cfg.vocab_size)
    tokenizer.save("bpe_tokenizer.json")
    print("Tokenizer training complete.")

    processor = partial(process_classification_data, tokenizer=tokenizer, max_len=cfg.max_len)
    dataset = GenericNLPDataset(df_sample, processor)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    if val_size == 0 and train_size > 0:
        train_dataset = dataset
        val_dataset = dataset
    else:
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    if cfg.model.name == 'bert':
        backbone = BERT(
            vocab_size=len(tokenizer.token_to_id),
            embed_size=cfg.model.embed_size,
            num_layers=cfg.model.num_layers,
            heads=cfg.model.heads,
            ff_hidden_dim=cfg.model.ff_hidden_dim,
            dropout=cfg.model.dropout
        )
        embed_size = cfg.model.embed_size
    else:
        raise ValueError(f"Unsupported model type: {cfg.model.name}")

    model = TextClassifier(backbone=backbone, embed_size=embed_size, num_classes=cfg.num_classes)

    print("Starting training with Hydra configuration...")
    train_model(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=cfg.training.epochs,
        learning_rate=cfg.training.learning_rate,
        batch_size=cfg.training.batch_size,
        save_path=cfg.training.save_path,
        task_type='classification'
    )
    print("Training complete.")

if __name__ == '__main__':
    main()