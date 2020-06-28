import torch
import argparse
from yacs.config import CfgNode as CN
import numpy as np
from data import get_data_iter
from model import TextCNN
from utils import log_info, get_load_path

torch.manual_seed(0)


def main():
    _, _, test_dataiter, embedding_vectors = get_data_iter(cfg)
    model = TextCNN(cfg, embedding_vectors, cfg.LOAD_PATH)
    # model = TextCNN(cfg, embedding_vectors)
    if cfg.CUDA:
        model.cuda()
    evaluate('test', model, test_dataiter, cfg.CUDA, cfg.SHOW_MISTAKES)


def evaluate(split, model, eval_dataiter, use_cuda, show_mistakes=False):
    model.eval()
    for batch in eval_dataiter:
        feature, target = batch.text, batch.label
        feature = feature.data.t()
        if use_cuda:
            feature = feature.cuda()
        with torch.no_grad():
            print(feature)
            logits = model(feature)
        pred = torch.argmax(logits, dim=1)
        pred, target = pred.cpu().numpy(), target.numpy()
        tp = np.logical_and(pred == 1, pred == target).sum()
        tn = np.logical_and(pred == 0, pred == target).sum()
        fp = np.logical_and(pred == 1, pred != target).sum()
        fn = np.logical_and(pred == 0, pred != target).sum()
        acc = (tp+tn) / (tp+tn+fp+fn)
        precision = tp / (tp+fp)
        recall = tp / (tp+fn)
        f1 = 2 * (precision * recall) / (precision + recall)
        if show_mistakes:
            wrong_idx = np.where(pred != target)[0]
            print('Incorrectly Classified Texts:')
            for idx in wrong_idx:
                print(f'{pred[idx]}({target[idx]})', end=' ')
                itos = eval_dataiter.dataset.fields['text'].vocab.itos
                for i in feature[idx].cpu().detach().numpy():
                    if i != 1:
                        print(itos[i], end='')
                print()
    log_info(f"Eval on {split}: Acc {acc}, Precision {precision}, Recall {recall}, F1 {f1} (TP {tp}, TN {tn}, FP {fp}, FN {fn})")
    return acc, precision, recall, f1, tp, tn, fp, fn


if __name__ == '__main__':
    cfg = CN()
    cfg.CUDA = torch.cuda.is_available()
    cfg.DATASET_NAME = 'movie'
    cfg.FILTER_NUM = 100
    cfg.FILTER_SIZE = [3, 4, 5]
    cfg.EMBEDDING_DIM = 128
    cfg.PRETRAINED_EMBEDDING = True
    cfg.PRETRAINED_PATH = 'pretrained/sgns.zhihu.word'
    cfg.FINETUNE_EMBEDDING = False
    cfg.MULTICHANNEL = False  # use 2 channels of word embedding
    cfg.DROPOUT_RATE = 0.5
    cfg.EXPERIMENT_NAME = f'multichannel'
    cfg.SHOW_MISTAKES = True
    # ---------
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("opts", help="Modify configs using the command-line", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cfg.merge_from_list(args.opts)
    # ---------
    if cfg.EXPERIMENT_NAME == 'baseline':
        cfg.PRETRAINED_EMBEDDING = False
        cfg.FINETUNE_EMBEDDING = False
        cfg.MULTICHANNEL = False
    elif cfg.EXPERIMENT_NAME == 'pretrain':
        cfg.PRETRAINED_EMBEDDING = True
        cfg.FINETUNE_EMBEDDING = False
        cfg.MULTICHANNEL = False
    elif cfg.EXPERIMENT_NAME == 'pretrain_finetune':
        cfg.PRETRAINED_EMBEDDING = True
        cfg.FINETUNE_EMBEDDING = True
        cfg.MULTICHANNEL = False
    elif cfg.EXPERIMENT_NAME == 'multichannel':
        cfg.PRETRAINED_EMBEDDING = True
        cfg.FINETUNE_EMBEDDING = True
        cfg.MULTICHANNEL = True
    else:
        raise NotImplementedError
    cfg.BATCH_SIZE = 64  # useless
    cfg.EXPERIMENT_NAME += f'_{cfg.DATASET_NAME}'
    cfg.LOAD_PATH = get_load_path(cfg.EXPERIMENT_NAME)
    print(cfg)
    main()