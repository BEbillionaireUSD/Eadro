import os
import time
import copy

import torch
from torch import nn
import logging

from model import MainModel
from sklearn.metrics import ndcg_score

class BaseModel(nn.Module):
    def __init__(self, event_num, metric_num, node_num, device, lr=1e-3, epoches=50, patience=5, result_dir='./', hash_id=None, **kwargs):
        super(BaseModel, self).__init__()
        
        self.epoches = epoches
        self.lr = lr
        self.patience = patience # > 0: use early stop
        self.device = device

        self.model_save_dir = os.path.join(result_dir, hash_id)
        self.model = MainModel(event_num, metric_num, node_num, device, **kwargs)
        self.model.to(device)
    
    def evaluate(self, test_loader, datatype="Test"):
        self.model.eval()
        hrs, ndcgs = np.zeros(5), np.zeros(5)
        TP, FP, FN = 0, 0, 0
        batch_cnt, epoch_loss = 0, 0.0 
        
        with torch.no_grad():
            for graph, ground_truths in test_loader:
                res = self.model.forward(graph.to(self.device), ground_truths)
                for idx, faulty_nodes in enumerate(res["y_pred"]):
                    culprit = ground_truths[idx].item()
                    if culprit == -1:
                        if faulty_nodes[0] == -1: TP+=1
                        else: FP += 1
                    else:
                        if faulty_nodes[0] == -1: FN+=1
                        else: 
                            TP+=1
                            rank = list(faulty_nodes).index(culprit)
                            for j in range(5):
                                hrs[j] += int(rank <= j)
                                ndcgs[j] += ndcg_score([res["y_prob"][idx]], [res["pred_prob"][idx]], k=j+1)
                epoch_loss += res["loss"].item()
                batch_cnt += 1
        
        pos = TP+FN
        eval_results = {
                "F1": TP*2.0/(TP+FP+pos) if (TP+FP+pos)>0 else 0,
                "Rec": TP*1.0/pos if pos > 0 else 0,
                "Pre": TP*1.0/(TP+FP) if (TP+FP) > 0 else 0}
        
        for j in [1, 3, 5]:
            eval_results["HR@"+str(j)] = hrs[j-1]*1.0/pos
            eval_results["ndcg@"+str(j)] = ndcgs[j-1]*1.0/pos
            
        logging.info("{} -- {}".format(datatype, ", ".join([k+": "+str(f"{v:.4f}") for k, v in eval_results.items()])))

        return eval_results
    
    def fit(self, train_loader, test_loader=None, evaluation_epoch=10):
        best_hr1, coverage, best_state, eval_res = -1, None, None, None # evaluation
        pre_loss, worse_count = float("inf"), 0 # early break

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        #optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.99)
        
        for epoch in range(1, self.epoches+1):
            self.model.train()
            batch_cnt, epoch_loss = 0, 0.0
            epoch_time_start = time.time()
            for graph, label in train_loader:
                optimizer.zero_grad()
                loss = self.model.forward(graph.to(self.device), label)['loss']
                loss.backward()
                # if self.debug:
                #     for name, parms in self.model.named_parameters():
                #         if name=='encoder.graph_model.net.weight':
                #             print(name, "--> grad:",parms.grad)
                optimizer.step()
                epoch_loss += loss.item()
                batch_cnt += 1
            epoch_time_elapsed = time.time() - epoch_time_start

            epoch_loss = epoch_loss / batch_cnt
            logging.info("Epoch {}/{}, training loss: {:.5f} [{:.2f}s]".format(epoch, self.epoches, epoch_loss, epoch_time_elapsed))

            ####### early break #######
            if epoch_loss > pre_loss:
                worse_count += 1
                if self.patience > 0 and worse_count >= self.patience:
                    logging.info("Early stop at epoch: {}".format(epoch))
                    break
            else: worse_count = 0
            pre_loss = epoch_loss

            ####### Evaluate test data during training #######
            if (epoch+1) % evaluation_epoch == 0:
                test_results = self.evaluate(test_loader, datatype="Test")
                if test_results["HR@1"] > best_hr1:
                    best_hr1, eval_res, coverage  = test_results["HR@1"], test_results, epoch
                    best_state = copy.deepcopy(self.model.state_dict())

                self.save_model(best_state)
            
        if coverage > 5:
            logging.info("* Best result got at epoch {} with HR@1: {:.4f}".format(coverage, best_hr1))
        else:
            logging.info("Unable to convergence!")

        return eval_res, coverage
    
    def load_model(self, model_save_file=""):
        self.model.load_state_dict(torch.load(model_save_file, map_location=self.device))

    def save_model(self, state, file=None):
        if file is None: file = os.path.join(self.model_save_dir, "model.ckpt")
        try:
            torch.save(state, file, _use_new_zipfile_serialization=False)
        except:
            torch.save(state, file)
