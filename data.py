import torchtext.data as data
import random
random.seed(0)
from torchtext.vocab import Vectors


def get_data_iter(cfg):
    dataset_name = cfg.DATASET_NAME
    text_field = data.Field(lower=True)
    label_field = data.Field(sequential=False, use_vocab=False)
    train_dataset, val_dataset, test_dataset = data.TabularDataset.splits(
        path=f'./dataset/{dataset_name}', train=f'{dataset_name}_train.csv',
        validation=f'{dataset_name}_val.csv', test=f'{dataset_name}_test.csv', format='csv',
        fields=[('label', label_field), ('text', text_field)])
    if cfg.PRETRAINED_EMBEDDING:
        print('loading?')
        vectors = Vectors(name=cfg.PRETRAINED_PATH)
        text_field.build_vocab(train_dataset, val_dataset, test_dataset, vectors=vectors)
        cfg.EMBEDDING_DIM = text_field.vocab.vectors.size()[-1]
    else:
        text_field.build_vocab(train_dataset, val_dataset, test_dataset)
    label_field.build_vocab(train_dataset, val_dataset, test_dataset)
    cfg.VOCABULARY_SIZE = len(text_field.vocab)

    train_dataiter, val_dataiter, test_dataiter = data.Iterator.splits(
        (train_dataset, val_dataset, test_dataset),
        batch_sizes=(cfg.BATCH_SIZE, len(val_dataset), len(test_dataset)),
        sort_key=lambda x: len(x.text),
        repeat=False,
        shuffle=True)
    return train_dataiter, val_dataiter, test_dataiter, text_field.vocab.vectors


def load_word_vectors(model_fname):
    vectors = Vectors(name=model_fname)
    return vectors