import os
import time
import torch
import argparse
import torch.nn.functional as F
from yacs.config import CfgNode as CN
import tqdm
from data import get_data_iter
from model import TextCNN
from utils import handle_keyboard_interruption, handle_other_exception, setup_logger, log_info
torch.manual_seed(0)

cfg = CN()
cfg.CUDA = torch.cuda.is_available()
cfg.BATCH_SIZE = 64
cfg.DATASET_NAME = 'flight'
cfg.FILTER_NUM = 100
cfg.FILTER_SIZE = [3, 4, 5]
cfg.EMBEDDING_DIM = 128
cfg.PRETRAINED_EMBEDDING = True
cfg.PRETRAINED_PATH = 'pretrained/sgns.zhihu.word'
cfg.FINETUNE_EMBEDDING = False
cfg.MULTICHANNEL = False    # use 2 channels of word embedding
cfg.DROPOUT_RATE = 0.5
cfg.TOTAL_EPOCHS = 256
cfg.LR = 0.001
cfg.EXPERIMENT_NAME = f'pretrain'

# ---------
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("opts", help="Modify configs using the command-line", default=None, nargs=argparse.REMAINDER)
args = parser.parse_args()
cfg.merge_from_list(args.opts)
# ---------
cfg.EXPERIMENT_NAME += f'_{cfg.DATASET_NAME}'
cfg.OUTPUT_DIR = f'outputs/{cfg.EXPERIMENT_NAME}'

os.makedirs('outputs/', exist_ok=True)
setup_logger(cfg, 't')


def main():
    train_dataiter, val_dataiter, test_dataiter, embedding_vectors = get_data_iter(cfg)
    model = TextCNN(cfg, embedding_vectors)
    if cfg.CUDA:
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR)

    best_acc = 0
    for epoch in range(1, cfg.TOTAL_EPOCHS + 1):
        model.train()
        for batch_i, batch in enumerate(train_dataiter):
            start_time = time.time()
            feature, target = batch.text, batch.label
            feature = feature.data.t()
            # target = target.data.sub(1)
            if cfg.CUDA:
                feature, target = feature.cuda(), target.cuda()
            optimizer.zero_grad()
            logits = model(feature)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()
            corrects_num = (torch.max(logits, 1)[1].view(target.size()).data == target.data).sum()
            train_acc = 100.0 * corrects_num / batch.batch_size
            log_info(f"Epoch {epoch}/{cfg.TOTAL_EPOCHS}, Batch {batch_i}/{len(train_dataiter)}, Loss {loss.data}, "
                     f"Train Acc {train_acc}({corrects_num}/{batch.batch_size}), Time {time.time()-start_time}s")
            break
        eval_acc = evaluate(cfg, model, val_dataiter)
        if best_acc < eval_acc:
            best_acc = eval_acc
            save_model_weights(model, cfg)
    evaluate(cfg, model, test_dataiter)


def evaluate(cfg, model, eval_dataiter):
    model.eval()
    corrects_num = 0
    for batch in tqdm.tqdm(eval_dataiter, desc='EVALUATING'):
        feature, target = batch.text, batch.label
        feature = feature.data.t()
        # target = target.data.sub(1)
        if cfg.CUDA:
            feature, target = feature.cuda(), target.cuda()
        logits = model(feature)
        corrects_num += (torch.max(logits, 1)[1].view(target.size()).data == target.data).sum()
    size = len(eval_dataiter.dataset)
    accuracy = 100.0 * corrects_num / size
    log_info(f"EVALUATION:\nACC {accuracy}({corrects_num}/{size})")
    return accuracy


def save_model_weights(model, cfg):
    torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, f'model_best.pth'))


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        handle_keyboard_interruption(cfg)
    except:
        handle_other_exception(cfg)