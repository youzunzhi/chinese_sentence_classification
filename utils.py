import torch
import logging
import os, sys
import datetime
import shutil


def handle_keyboard_interruption(cfg):
    assert cfg.OUTPUT_DIR.find('[') != -1
    save = input('save the log files(%s)?[y|n]' % cfg.OUTPUT_DIR)
    if save == 'y':
        print('log files may be saved in', cfg.OUTPUT_DIR)
    elif save == 'n':
        shutil.rmtree(cfg.OUTPUT_DIR)
        print('log directory removed:', cfg.OUTPUT_DIR)
    else:
        print('unknown input, saved by default')


def handle_other_exception(cfg):
    import traceback
    print(traceback.format_exc())
    assert cfg.OUTPUT_DIR != 'runs/'
    assert cfg.OUTPUT_DIR.find('[') != -1
    print('log directory removed:', cfg.OUTPUT_DIR)
    shutil.rmtree(cfg.OUTPUT_DIR)


def setup_logger(cfg, log_prefix, distributed_rank=0):
    # ---- make output dir ----
    # each experiment's output is in the dir named after the time when it starts to run
    if torch.cuda.is_available():
        log_dir_name = log_prefix + '-[{}]'.format(
            (datetime.datetime.now() + datetime.timedelta(hours=8)).strftime('%m%d%H%M%S'))
    else:
        log_dir_name = log_prefix + '-[{}]'.format((datetime.datetime.now()).strftime('%m%d%H%M%S'))
    log_dir_name += cfg.EXPERIMENT_NAME
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, log_dir_name)
    os.mkdir(cfg.OUTPUT_DIR)

    # ---- set up logger ----
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s: %(message)s", '%m%d%H%M%S')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    txt_name = 'log.txt'
    fh = logging.FileHandler(os.path.join(cfg.OUTPUT_DIR, txt_name), mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def log_info(log_str):
    logging.getLogger().info(log_str)


def get_load_path(experiment_name):
    def recursive_glob(rootdir=".", suffix=""):
        """Performs recursive glob with given suffix and rootdir
            :param rootdir is the root directory
            :param suffix is the suffix to be searched
        """
        return [
            os.path.join(looproot, filename)
            for looproot, _, filenames in os.walk(rootdir)
            for filename in filenames
            if filename.endswith(suffix)
        ]

    return recursive_glob(f'outputs/{experiment_name}', 'pth')[0]


def make_csv(dataset, split):
    ori_path = f'dataset/{dataset}/{dataset}_{split}'
    with open(ori_path, 'r') as fr:
        with open(f'dataset/{dataset}/{dataset}_{split}.csv', 'w') as f:
            for l in fr.readlines():
                write_line = l.replace(',', '，')
                write_line = write_line[0] + ',' + write_line[2:]
                write_line = write_line.replace('"', '')
                f.write(write_line)


"""
for d in ['flight', 'laptop', 'movie']:
    for s in ['train', 'val', 'test']:
        make_csv(d, s)
"""
