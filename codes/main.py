from torch.utils.data import Dataset, DataLoader
import torch
import dgl
class chunkDataset(Dataset): #[node_num, T, else]
    def __init__(self, chunks, node_num, edges):
        self.data = []
        self.idx2id = {}
        for idx, chunk_id in enumerate(chunks.keys()):
            self.idx2id[idx] = chunk_id
            chunk = chunks[chunk_id]
            graph = dgl.graph(edges, num_nodes=node_num)
            graph.ndata["logs"] = torch.FloatTensor(chunk["logs"])
            graph.ndata["metrics"] = torch.FloatTensor(chunk["metrics"])
            graph.ndata["traces"] = torch.FloatTensor(chunk["traces"])
            self.data.append((graph, chunk["culprit"]))
                
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]
    def __get_chunk_id__(self, idx):
        return self.idx2id[idx]

from utils import *
from base import BaseModel

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--random_seed", default=42, type=int)

### Training params
parser.add_argument("--gpu", default=True, type=lambda x: x.lower() == "true")
parser.add_argument("--epoches", default=50, type=int)
parser.add_argument("--batch_size", default=256, type=int)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--patience", default=10, type=int)

##### Fuse params
parser.add_argument("--self_attn", default=True, type=lambda x: x.lower() == "true")
parser.add_argument("--fuse_dim", default=128, type=int)
parser.add_argument("--alpha", default=0.5, type=float)
parser.add_argument("--locate_hiddens", default=[64], type=int, nargs='+')
parser.add_argument("--detect_hiddens", default=[64], type=int, nargs='+')

##### Source params
parser.add_argument("--log_dim", default=16, type=int)
parser.add_argument("--trace_kernel_sizes", default=[2], type=int, nargs='+')
parser.add_argument("--trace_hiddens", default=[64], type=int, nargs='+')
parser.add_argument("--metric_kernel_sizes", default=[2], type=int, nargs='+')
parser.add_argument("--metric_hiddens", default=[64], type=int, nargs='+')
parser.add_argument("--graph_hiddens", default=[64], type=int, nargs='+')
parser.add_argument("--attn_head", default=4, type=int, help="For gat or gat-v2")
parser.add_argument("--activation", default=0.2, type=float, help="use LeakyReLU, shoule be in (0,1)")

##### Data params
parser.add_argument("--data", type=str, required=True)
parser.add_argument("--result_dir", default="../result/")

params = vars(parser.parse_args())

import logging
def get_device(gpu):
    if gpu and torch.cuda.is_available():
        logging.info("Using GPU...")
        return torch.device("cuda")
    logging.info("Using CPU...")
    return torch.device("cpu")
    

def collate(data):
    graphs, labels = map(list, zip(*data))
    batched_graph = dgl.batch(graphs)
    return batched_graph , torch.tensor(labels)

def run(evaluation_epoch=10):
    data_dir = os.path.join("../chunks", params["data"])

    metadata = read_json(os.path.join(data_dir, "metadata.json"))
    event_num, node_num, metric_num =  metadata["event_num"], metadata["node_num"], metadata["metric_num"]
    params["chunk_lenth"] = metadata["chunk_lenth"]

    hash_id = dump_params(params)
    params["hash_id"] = hash_id
    seed_everything(params["random_seed"])
    device = get_device(params["gpu"])

    train_chunks, test_chunks = load_chunks(data_dir)

    train_data = chunkDataset(train_chunks, node_num)
    test_data = chunkDataset(test_chunks, node_num)
    
    train_dl = DataLoader(train_data, batch_size=params["batch_size"], shuffle=True, collate_fn=collate, pin_memory=True)
    test_dl = DataLoader(test_data, batch_size=params["batch_size"], shuffle=False, collate_fn=collate, pin_memory=True)

    model = BaseModel(event_num, metric_num, node_num, device, **params)
    scores, converge = model.fit(train_dl, test_dl, evaluation_epoch=evaluation_epoch)

    dump_scores(params["result_dir"], hash_id, scores, converge)
    logging.info("Current hash_id {}".format(hash_id))

if "__main__" == __name__:
    run()
