import string
import random
chunkids = set()
src = string.ascii_letters + string.digits
def get_chunkid():
    while(True):
        chunkid = random.sample(src, 8)
        random.shuffle(chunkid)
        chunkid = ''.join(chunkid)
        #print(chunkid)
        if chunkid not in chunkids:
            chunkids.add(chunkid)
            return chunkid

from util import *
def get_basic(info, idx, name, chunk_lenth=10, threshold=1, **kwargs):
    records = read_json(os.path.join("./parsed_data", name, "records"+idx+".json"))
    faults = [(f_record["s"], f_record["e"], info.service2nid[f_record["service"]])
                for f_record in records["faults"]]
    start, end = records["start"], records["end"]
    
    ########## Generate annotated intervals ##########
    intervals = [(s, s+chunk_lenth-1) 
                 for s in range(start, end-chunk_lenth+1)]
    labels = [-1]*len(intervals)
    for chunk_idx, (s, e) in enumerate(intervals):
        for (fs, fe, culprit) in faults:
            overlap = 0
            if (s>=fs and s<=fe): overlap = fe-s+1
            elif (e>=fs and e<=fe): overlap = e-fs+1
            if overlap >= threshold: labels[chunk_idx] = culprit
            if overlap > 0: break
    
    print('# starts at {}/{} and ends at {}/{}'.format(intervals[0][0], start, intervals[-1][-1], end))
    return intervals, labels

import os
import pickle
from collections import defaultdict

from single_process import deal_logs, deal_traces, deal_metrics
def get_chunks(info, idx, name, chunk_lenth=10, **kwargs):
    intervals, labels = get_basic(info, idx, name=name, chunk_lenth=chunk_lenth, **kwargs)
    
    aim_dir = os.path.join("../chunks", name, idx)
    if not os.path.exists(aim_dir): os.mkdir(aim_dir)
    if os.path.exists(os.path.join(aim_dir, "traces.pkl")):
        with open(os.path.join(aim_dir, "traces.pkl"), "rb") as fr: 
            traces = pickle.load(fr)
    else: traces = deal_traces(intervals, info, idx, name=name, chunk_lenth=chunk_lenth)
    if os.path.exists(os.path.join(aim_dir, "metrics.pkl")):
        with open(os.path.join(aim_dir, "metrics.pkl"), "rb") as fr: 
            metrics = pickle.load(fr)
    else: metrics = deal_metrics(intervals, info, idx, name=name, chunk_lenth=chunk_lenth)
    if os.path.exists(os.path.join(aim_dir, "logs.pkl")):
        with open(os.path.join(aim_dir, "logs.pkl"), "rb") as fr: 
            logs = pickle.load(fr)
    else: logs = deal_logs(intervals, info, idx, name=name)

    print("*** Aligning multi-source data...")
    chunks = defaultdict(dict)
    for idx in range(len(intervals)):
        chunk_id = get_chunkid()
        chunks[chunk_id]["traces"] = traces["latency"][idx] #[node_num, chunk_lenth, 2]
        chunks[chunk_id]["metrics"] = metrics[idx]
        chunks[chunk_id]["logs"] = logs[idx]
        chunks[chunk_id]['culprit'] = labels[idx]

    return chunks

import numpy as np
def get_all_chunks(name, chunk_lenth=10, **kwargs):
    aim_dir = os.path.join("../chunks", name)
    if not os.path.exists(aim_dir): os.mkdir(aim_dir)
    ############## Concat all chunks ##############
    bench = "trainticket" if name[0] == "A" else "socialnetwork" 
    info = Info(bench)
    print('# Node num:', info.node_num)
  
    chunks = {}
    idx = 0
    while True:
        if os.path.exists(os.path.join("./parsed_data", name, "records"+str(idx)+".json")):
            print("\n\n", "^"*20, "Now dealing with", idx, "^"*20)
            new_chunks = get_chunks(info, str(idx), chunk_lenth=chunk_lenth, name=name, **kwargs)
            chunks.update(new_chunks)
            idx += 1
        else:
            break
    print("# Data Genenaration Batch Size: ", idx)
    with open(os.path.join(aim_dir, "chunks.pkl"), "wb") as fw:
        pickle.dump(chunks, fw)
    
    ############## Update info for metadata ##############
    info.add_info("chunk_lenth", chunk_lenth)
    info.add_info("chunk_num", len(chunks))
    info.add_info("edges", info.edges)
    info.add_info("event_num", chunks[list(chunks.keys())[0]]["logs"].shape[-1])
    
    if os.path.exists(os.path.join(aim_dir, "metadata.json")):
        os.remove(os.path.join(aim_dir, "metadata.json"))
    json_pretty_dump(info.metadata, os.path.join(aim_dir, "metadata.json"))

    return chunks

def split_chunks(name, test_ratio=0.2, concat=False, **kwargs):
    if concat:
        print("*** Concating chunks...")
        chunk_num = 0
        chunks = {}
        for dir in os.listdir("../chunks"):
            if not dir.startswith(name[0]): continue
            if os.path.exists(os.path.join("../chunks", dir, "chunks.pkl")):
                with open(os.path.join("../chunks", dir, "chunks.pkl"), "rb") as fr: 
                    chunks.update(pickle.load(fr))
                metadata = read_json(os.path.join("../chunks", dir, "metadata.json"))
                chunk_num += metadata["chunk_num"]
        os.makedirs(os.path.join("../chunks", name[0]), exist_ok=True)
        metadata["chunk_num"] = chunk_num
        json_pretty_dump(metadata, os.path.join("../chunks", name[0], "metadata.json"))

    elif os.path.exists(os.path.join("../chunks", name, "chunks.pkl")):
        with open(os.path.join("../chunks", name, "chunks.pkl"), "rb") as fr: 
            chunks.update(pickle.load(fr))
    else: chunks = get_all_chunks(name=name, **kwargs)

    print("\n *** Spliting chunks into training and testing sets...")
    
    chunk_num = len(chunks)
    chunk_hashids = np.array(list(chunks.keys()))
    chunk_idx = list(range(chunk_num))

    train_num = int((1 - test_ratio) * chunk_num)
    test_num = int(test_ratio * chunk_num)
    np.random.shuffle(chunk_idx)

    train_idx = chunk_idx[:train_num]
    test_idx = chunk_idx[train_num:train_num+test_num]

    train_chunks = {k:chunks[k] for k in chunk_hashids[train_idx]}
    test_chunks = {k:chunks[k] for k in chunk_hashids[test_idx]}

    aim = name[0] if concat else name
    with open(os.path.join("../chunks", aim, "chunk_train.pkl"), "wb") as fw:
        pickle.dump(train_chunks, fw)
    with open(os.path.join("../chunks", aim, "chunk_test.pkl"), "wb") as fw:
        pickle.dump(test_chunks, fw)

    ########## Print statistics ##########
    label_count = {}
    for _, v in chunks.items(): 
        label = v['culprit']
        if label not in label_count: label_count[label] = 0
        label_count[label] += 1

    train_labels = [v['culprit']!=-1 for _, v in train_chunks.items()]
    test_labels = [v['culprit']!=-1 for _, v in test_chunks.items()]

    print("# train chunks: {}/{} ({:.4f}%)".format(sum(train_labels), train_num, 100*(sum(train_labels)/train_num)))
    print("# test chunks: {}/{} ({:.4f}%)".format(sum(test_labels), test_num, 100*(sum(test_labels)/test_num)))
    print("# total chunk: {}/{} ({:.4f}%)".format(sum(train_labels)+sum(test_labels), chunk_num,  100*((sum(train_labels)+sum(test_labels))/chunk_num)))
    for label in sorted(list(label_count.keys())):
        if label > -1:
            print('Node {} have {} faulty chunks'.format(label, label_count[label]))


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--concat", action="store_true")
parser.add_argument("--delete_all", action="store_true")
parser.add_argument("--delete", action="store_true", help="just remove the final chunks and retain pre-processed data")
parser.add_argument("--threshold", default=1, type=int)
parser.add_argument("--chunk_lenth", default=10, type=int)
parser.add_argument("--test_ratio", default=0.3, type=float)
parser.add_argument("--name", required=True, help="The system name")
params = vars(parser.parse_args())

if "__main__" == __name__:
    aim_dir = os.path.join("../chunks", params['name'])
    if params['delete_all']:
        _input = input("Do you really want to delete all previous files?! Input yes if you are so confident.\n")
        flag = (_input.lower() == 'yes')
        if flag and os.path.exists(aim_dir) and len(aim_dir)>2:
            import shutil
            shutil.rmtree(aim_dir)
        else:
            print("Thank you for thinking twice!")
            exit()
    if params['delete'] and os.path.exists(os.path.join(aim_dir, "chunks.pkl")):
        os.remove(os.path.join(aim_dir, "chunks.pkl"))
    split_chunks(**params)

    

   





    





        


    


    




