import os
import time
import torch
import argparse
import torch.nn.functional as F
from yacs.config import CfgNode as CN
import tqdm
from data import get_data_iter
from model import TextCNN
from utils import handle_keyboard_interruption, handle_other_exception, setup_logger, log_info, get_load_path
torch.manual_seed(0)
cfg = CN()
cfg.CUDA = torch.cuda.is_available()
cfg.DATASET_NAME = 'movie'
cfg.EMBEDDING_DIM = 128
cfg.PRETRAINED_EMBEDDING = True
cfg.PRETRAINED_PATH = 'pretrained/sgns.zhihu.word'
cfg.FINETUNE_EMBEDDING = False
cfg.MULTICHANNEL = False    # use 2 channels of word embedding
cfg.DROPOUT_RATE = 0.5
cfg.EXPERIMENT_NAME = f''
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
cfg.BATCH_SIZE = 64     # useless
cfg.EXPERIMENT_NAME += f'_{cfg.DATASET_NAME}'
cfg.LOAD_PATH = get_load_path(cfg.EXPERIMENT_NAME)


def main():
    _, _, test_dataiter, embedding_vectors = get_data_iter(cfg)
    model = TextCNN(cfg, embedding_vectors)
    if cfg.CUDA:
        model.cuda()
    model.eval()
    corrects_num = 0
    for batch in tqdm.tqdm(test_dataiter, desc='EVALUATING'):
        feature, target = batch.text, batch.label
        feature = feature.data.t()
        if cfg.CUDA:
            feature, target = feature.cuda(), target.cuda()
        logits = model(feature)
        wrong_idx = torch.where(torch.max(logits, 1)[1].view(target.size()).data != target.data)[0]
        print('Incorrectly Classified Texts:')
        for i in wrong_idx.numpy():
            for t in test_dataiter.dataset.examples[i].text:
                print(t, end='')
            print()
        corrects_num += (torch.max(logits, 1)[1].view(target.size()).data == target.data).sum()
    size = len(test_dataiter.dataset)
    accuracy = 100.0 * corrects_num / size
    log_info(f"Eval on test: ACC {accuracy}({corrects_num}/{size})")
    return accuracy


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        handle_keyboard_interruption(cfg)
    except:
        handle_other_exception(cfg)