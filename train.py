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
from eval import evaluate
torch.manual_seed(0)

cfg = CN()
cfg.CUDA = torch.cuda.is_available()
cfg.BATCH_SIZE = 64
cfg.DATASET_NAME = 'laptop'
# ---- Model Definition ----
cfg.FILTER_NUM = 100
cfg.FILTER_SIZE = [3, 4, 5]
cfg.EMBEDDING_DIM = 128
cfg.DROPOUT_RATE = 0.5
# ---- Model Variation ----
cfg.PRETRAINED_EMBEDDING = True
cfg.PRETRAINED_PATH = 'pretrained/sgns.zhihu.word'
cfg.FINETUNE_EMBEDDING = False
cfg.MULTICHANNEL = False    # use 2 channels of word embedding
# ---- Training Scheme ----
cfg.TOTAL_EPOCHS = 50
cfg.LR = 0.001
# ---------
cfg.EXPERIMENT_NAME = f'pretrain'
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
cfg.EXPERIMENT_NAME += f'_{cfg.DATASET_NAME}'
cfg.OUTPUT_DIR = f'outputs/{cfg.EXPERIMENT_NAME}'

os.makedirs('outputs/', exist_ok=True)
setup_logger(cfg, 't')
log_info(cfg)


def main():
    train_dataiter, val_dataiter, test_dataiter, embedding_vectors = get_data_iter(cfg)
    model = TextCNN(cfg, embedding_vectors)
    if cfg.CUDA:
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR)

    best_acc = 0
    not_improving_epochs = 0
    for epoch in range(1, cfg.TOTAL_EPOCHS + 1):
        model.train()
        for batch_i, batch in enumerate(train_dataiter):
            start_time = time.time()
            feature, target = batch.text, batch.label
            feature = feature.data.t()
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
        val_acc = evaluate('val', model, val_dataiter, cfg.CUDA)[0]
        if best_acc < val_acc:
            best_acc = val_acc
            save_model_weights(model, cfg, epoch)
            best_test_performace = evaluate('test', model, test_dataiter, cfg.CUDA)
            not_improving_epochs = 0
        elif not_improving_epochs >= 10:
            log_info('Early stop.')
            break
        else:
            not_improving_epochs += 1
    b = best_test_performace
    log_info(f"{cfg.EXPERIMENT_NAME} Best model: \n"
             f"Acc {b[0]:.4f}, Precision {b[1]:.4f}, Recall {b[2]:.4f}, F1 {b[3]:.4f}\n"
             f"(TP {b[4]}, TN {b[5]}, FP {b[6]}, FN {b[7]})")


def save_model_weights(model, cfg, epoch):
    log_info(f'saving best model of epoch {epoch}')
    torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, f'model_best.pth'))


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        handle_keyboard_interruption(cfg)
    except:
        handle_other_exception(cfg)