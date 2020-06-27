import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, cfg, embedding_vectors):
        super(TextCNN, self).__init__()
        self.cfg = cfg

        class_num = 3
        channel_num = 1
        filter_num = cfg.FILTER_NUM
        filter_sizes = cfg.FILTER_SIZE

        vocabulary_size = cfg.VOCABULARY_SIZE
        embedding_dim = cfg.EMBEDDING_DIM
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim)
        if cfg.PRETRAINED_EMBEDDING:
            self.embedding = self.embedding.from_pretrained(embedding_vectors, freeze=not cfg.FINETUNE_EMBEDDING)
        if cfg.MULTICHANNEL:
            self.embedding2 = nn.Embedding(vocabulary_size, embedding_dim).from_pretrained(embedding_vectors, freeze=False)
            channel_num += 1
        else:
            self.embedding2 = None
        self.convs = nn.ModuleList(
            [nn.Conv2d(channel_num, filter_num, (size, embedding_dim)) for size in filter_sizes])
        self.dropout = nn.Dropout(cfg.DROPOUT_RATE)
        self.fc = nn.Linear(len(filter_sizes) * filter_num, class_num)

    def forward(self, x):
        if self.embedding2:
            x = torch.stack([self.embedding(x), self.embedding2(x)], dim=1)
        else:
            x = self.embedding(x)
            x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits
