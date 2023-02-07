import os
import logging
import pickle

def load_chunks(data_dir):
    logging.info("Load from {}".format(data_dir))
    with open(os.path.join(data_dir, "chunk_train.pkl"), "rb") as fr:
        chunk_train = pickle.load(fr)
    with open(os.path.join(data_dir, "chunk_test.pkl"), "rb") as fr:
        chunk_test = pickle.load(fr)
    return chunk_train, chunk_test

import json
def read_json(filepath):
    if os.path.exists(filepath):
        assert filepath.endswith('.json')
        with open(filepath, 'r') as f:
            return json.loads(f.read())
    else: 
        logging.raiseExceptions("File path "+filepath+" not exists!")
        return

def json_pretty_dump(obj, filename):
    with open(filename, "w") as fw:
        json.dump(obj,fw, sort_keys=True, indent=4, separators=(",", ": "), ensure_ascii=False)


from datetime import datetime, timedelta
def dump_scores(result_dir, hash_id, scores, converge):
    with open(os.path.join(result_dir, 'experiments.txt'), 'a+') as fw:
        fw.write(hash_id+': '+(datetime.now()+timedelta(hours=8)).strftime("%Y/%m/%d-%H:%M:%S")+'\n')
        fw.write("* Test result -- " + '\t'.join(["{}:{:.4f}".format(k, v) for k,v in scores.items()])+'\n')
        fw.write('Best score got at epoch: '+str(converge)+'\n')
        fw.write('{}{}'.format('='*40, '\n'))

import hashlib
def dump_params(params):
    hash_id = hashlib.md5(str(sorted([(k, v) for k, v in params.items()])).encode("utf-8")).hexdigest()[0:8]
    result_dir = os.path.join(params["result_dir"], hash_id)
    os.makedirs(result_dir, exist_ok=True)

    json_pretty_dump(params, os.path.join(result_dir, "params.json"))

    log_file = os.path.join(result_dir, "running.log")
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s P%(process)d %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return hash_id

import random    
import numpy as np
import torch
def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
