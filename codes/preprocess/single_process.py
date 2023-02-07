import numpy as np
from tick.hawkes import HawkesADM4

def logHaw(chunk_logs, end_time, event_num, decay=3, ini_intensity=0.2):
    model = HawkesADM4(decay)
    model.fit(chunk_logs, end_time, baseline_start=np.ones(event_num)*ini_intensity)
    return np.array(model.baseline)

from util import read_json, Info
import pandas as pd
import os
import pickle

def deal_logs(intervals, info, idx, name):
    print("*** Dealing with logs...")
    
    df = pd.read_csv(os.path.join("./parsed_data", name, "logs"+idx+".csv"))
    templates = read_json(os.path.join("./parsed_data", name, "templates.json"))
    event_num = len(templates)
    
    print("# Real Template Number:", event_num)
    event_num += 1 #0: unseen

    event2id = {temp:idx+1 for idx, temp in enumerate(templates)} #0: unseen
    event2id["Unseen"] = 0
    res = np.zeros((len(intervals), info.node_num, event_num))

    no_log_chunk = 0 
    for chunk_idx, (s, e) in enumerate(intervals):
        if (chunk_idx+1) % 100 == 0: 
            print("Computing Hawkes of chunk {}/{}".format(chunk_idx+1, len(intervals)))
        try:
            rows = df.loc[(df["timestamp"] >=s) & (df["timestamp"]<=e)]
        except:
            no_log_chunk +=1
            continue

        service_events = rows.groupby("service")
        for service, sgroup in service_events:
            events = sgroup.groupby("event")
            knots = [np.array([0.0]) for _ in range(event_num)]
            for event, egroup in events:
                eid = 0 if event not in event2id else event2id[event]
                tmp = np.array(sorted(egroup["timestamp"].values))-s
                adds = np.array([idx*(1e-5) for idx in range(len(tmp))]) #In case of too many identical numbers
                knots[eid] = tmp+adds
            paras = logHaw(knots, end_time=e+1, event_num=event_num)
            res[chunk_idx, info.service2nid[service], :] = paras
    
    print("# Empty log:", no_log_chunk)   
    with open(os.path.join("../chunks", name, idx, "logs.pkl"), "wb") as fw:
        pickle.dump(res, fw)
    return res

z_zero_scaler = lambda x: (x-np.mean(x)) / (np.std(x)+1e-8)

def deal_metrics(intervals, info, idx, name, chunk_lenth):
    print("*** Dealing with metrics...")
    metric_num = len(info.metric_names)
    metrics = np.zeros((len(intervals), info.node_num, chunk_lenth, metric_num))
    
    for nid, service in enumerate(info.service_names):
        df = pd.read_csv(os.path.join("./parsed_data", name, "metrics"+idx, service+'.csv'))
        df[info.metric_names] = df[info.metric_names].apply(z_zero_scaler)
        df.set_index(["timestamp"], inplace=True)
        for chunk_idx, (s,e) in enumerate(intervals):
            values = df.loc[s:e, :].to_numpy()
            assert values.shape == (chunk_lenth, metric_num), "{} shape in {}--{}".format(values.shape,s,e)
            metrics[chunk_idx, nid, :, :] = values
    
    with open(os.path.join("../chunks", name, idx, "metrics.pkl"), "wb") as fw:
        pickle.dump(metrics, fw)
    return metrics

def deal_traces(intervals, info, idx, name, chunk_lenth):
    """
    Input:
        intervals=[(s,e)], the chunks covers the period of [s, e].
    Return:
        a dict containing info for each interval:
        -- cell of invok list : a dict contains invocations inside the given time period == as invocation-based edge-level features
            {s-t:[lat1, lat2, ...]}
        -- cell of latency list: a dict contains a np.array [chunk_lenth] denoting the average latency (per time slot) for each node 
                                === as trace-based node-level features
            {nid:np.array([lat_1, ..., lat_tau, ..., lat_chunk_lenth}])}
    """
    print("*** Dealing with traces...")
    traces = read_json(os.path.join("./parsed_data", name, "traces"+idx+".json"))
    invocations = [] # the number of invocations
    latency = np.zeros((len(intervals), info.node_num, chunk_lenth, 2))

    for chunk_idx, (s, e) in enumerate(intervals):
        invok = {}
        slots = [t for t in range(s, e+1)]
        for i, ts in enumerate(slots):
            if str(ts) in traces.keys(): # spans exist in the i-th time slot
                spans = traces[str(ts)]
                tmp_node_lat = [[] for _ in range(info.node_num)]
                for k, lat_lst in spans.items():
                    if k not in invok: invok[k] = 0
                    invok[k] += len(lat_lst)
                    t_node = int(k.split('-')[-1])
                    tmp_node_lat[t_node].extend(lat_lst)
                for t_node in range(info.node_num):
                    if len(tmp_node_lat[t_node]) > 0:
                        latency[chunk_idx][t_node][i][0] = np.mean(tmp_node_lat[t_node])
        invocations.append(invok)
    
    for i in range(info.node_num):
        latency[:, i, :, 0] = z_zero_scaler(latency[:, i, :, 0])
    
    chunk_traces = {"invok": invocations, "latency": latency}
    with open(os.path.join("../chunks", name, idx, "traces.pkl"), "wb") as fw:
        pickle.dump(chunk_traces, fw)
    return chunk_traces