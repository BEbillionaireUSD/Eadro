from logging import raiseExceptions
import os
import json
class Info():
    def __init__(self, bench='TrainTicket'):
        
        if bench.lower() == 'trainticket':
            tmp_apiList = ['assurance', 'auth',  'basic', 'cancel', 'config', 'contacts', 'food-map', 'food', 'inside-payment', 
                       'notification', 'order-other', 'order', 'payment', 'preserve', 'price', 'route-plan', 'route', 'seat', 
                       'security', 'station', 'ticketinfo', 'train', 'travel-plan', 'travel', 'travel2', 'user', 'verification-code']
            apiList = ['ts-{}-service'.format(api) for api in tmp_apiList]
            edge_info = {
                    "preserve": ["preserve", "seat", "security", "food", "order", "ticketinfo", "travel", "contacts", "notification", "user", "station"],
                    "seat":["seat", "order", "config", "travel"],   
                    "cancel":["inside-payment", "order-other", "order"],
                    "security": ["security", "order-other", "order"],
                    "food":["travel", "food-map", "station"],
                    "travel": ["travel", "order", "ticketinfo", "train", "route"],
                    "inside-payment":["payment", "order"],
                    "ticketinfo":["ticketinfo", "basic"],
                    "basic":["basic", "route", "price", "train", "station"],
                    "order-other":["station"],
                    "order":["order", "station", "assurance"],
                    "auth":["auth", "verification-code"]
                    }
            self.edge_info = {'ts-{}-service'.format(k):['ts-{}-service'.format(vi) 
                                                        for vi in v] for k,v in edge_info.items()} 

        elif bench.lower() == 'socialnetwork':
            apiList = ['social-graph-service', 'compose-post-service', 'post-storage-service', 'user-timeline-service', 'url-shorten-service', 'user-service',
                       'media-service', 'text-service', 'unique-id-service', 'user-mention-service', 'home-timeline-service', "nginx-web-server"]
            self.edge_info = {
                "compose-post-service": ["compose-post-service", "home-timeline-service", "media-service", "post-storage-service", 
                                        "text-service", "unique-id-service", "user-service", "user-timeline-service"],
                "home-timeline-service":["home-timeline-service", "post-storage-service", "social-graph-service"],
                "post-storage-service": ["post-storage-service"],
                "social-graph-service": ["social-graph-service", "user-service"],
                "text-service": ["text-service", "url-shorten-service", "user-mention-service"],
                "user-service": ["user-service"],
                "user-timeline-service": ["user-timeline-service"],
                "nginx-web-server": ["compose-post-service", "home-timeline-service", "nginx-web-server", "social-graph-service", "user-service"]
            }
        else:
            raiseExceptions("Not Implemented yet {}".format(bench))
        
        self.metric_names = ['cpu_usage_system', 'cpu_usage_total', 'cpu_usage_user', 'memory_usage', 'memory_working_set', 'rx_bytes', 'tx_bytes']
       
        self.service_names = apiList
        self.service2nid = {s:idx for idx, s in enumerate(self.service_names)}
        self.node_num = len(self.service_names)
        
        self.metadata = {
            "node_num": self.node_num,
            "metric_num": len(self.metric_names),
        }
        self.__get_edges()
    
    def __get_edges(self):
        src, des = [], []
        for s, v in self.edge_info.items():
            sid = self.service2nid[s]
            for t in v:
                src.append(sid)
                des.append(self.service2nid[t])
        self.edges = [src, des]
    
    def add_info(self, key, value):
        self.metadata[key] = value

def read_json(filepath):
    if os.path.exists(filepath):
        assert filepath.endswith('.json')
        with open(filepath, 'r') as f:
            return json.loads(f.read())
    else: 
        raiseExceptions("File path "+filepath+" not exists!")
        return

def json_pretty_dump(obj, filename):
    with open(filename, "w") as fw:
        json.dump(obj,fw, sort_keys=True, indent=4, separators=(",", ": "), ensure_ascii=False)


