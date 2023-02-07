<img width="200" alt="截屏2022-09-19 下午9 50 34" src="https://user-images.githubusercontent.com/112700133/191033061-ea4a1671-26c7-4d52-b3ed-3495a2ae0292.png">

![](https://img.shields.io/badge/version-0.1-green.svg) 

****
Artifacts accompanying the paper *Eadro: An End-to-End Troubleshooting Framework for Microservices on Multi-source Data* published at ICSE 2023. 
This tool try to model the intra- and inter-dependencies between microservices for troubleshooting, enabling end-to-end anomaly detection and root cause localization.

<img width="400" alt="dependency_00" src="https://user-images.githubusercontent.com/112700133/191036446-d4cf8d07-bd4e-4452-a3e2-f7d4e9da0624.png">

## Data
Our data are at https://doi.org/10.5281/zenodo.7615393.

## Dependencies
`pip install -r requirements.txt`


## Architecture
![Eadro](https://user-images.githubusercontent.com/49298462/217256928-f0d61857-678b-4456-a024-359326a2c45d.png)

## Folder Structure
```
.
├── README.md
├── codes                                             
│   ├── base.py                                         traing and test
│   ├── main.py                                         lanuch the framework
│   ├── model.py                                        the main body (model) of the work
│   ├── preprocess                                      data preprocess                
│   │   ├── align.py                                    align different data sources according to the time
│   │   ├── single_process.py                           handle each data source individually
│   │   └── util.py
│   └── utils.py
├── data_demo                                           a small amount of data used in this paper
│   ├── TT.2022-04-19T001753D2022-04-19T020534
│   │   ├── logs.json
│   │   ├── metrics
│   │   │   ├── ts-assurance-service.csv
│   │   │   ├── ts-auth-service.csv
│   │   │   ├── ts-basic-service.csv
│   │   │   ├── ts-cancel-service.csv
│   │   │   ├── ts-config-service.csv
│   │   │   ├── ts-contacts-service.csv
│   │   │   ├── ts-food-map-service.csv
│   │   │   ├── ts-food-service.csv
│   │   │   ├── ts-inside-payment-service.csv
│   │   │   ├── ts-notification-service.csv
│   │   │   ├── ts-order-other-service.csv
│   │   │   ├── ts-order-service.csv
│   │   │   ├── ts-payment-service.csv
│   │   │   ├── ts-preserve-service.csv
│   │   │   ├── ts-price-service.csv
│   │   │   ├── ts-route-plan-service.csv
│   │   │   ├── ts-route-service.csv
│   │   │   ├── ts-seat-service.csv
│   │   │   ├── ts-security-service.csv
│   │   │   ├── ts-station-service.csv
│   │   │   ├── ts-ticketinfo-service.csv
│   │   │   ├── ts-train-service.csv
│   │   │   ├── ts-travel-plan-service.csv
│   │   │   ├── ts-travel-service.csv
│   │   │   ├── ts-travel2-service.csv
│   │   │   ├── ts-user-service.csv
│   │   │   └── ts-verification-code-service.csv
│   │   └── spans.json.zip
│   └── TT.fault-2022-04-19T001753D2022-04-19T020534.json
├── requirements.txt
└── structure.txt
```

## UI
The final visualized page should be like:
<img width="1919" alt="截屏2023-02-07 下午9 28 22" src="https://user-images.githubusercontent.com/49298462/217257747-e53afafe-ea3f-4024-8760-34d0963a863d.png">

## Concact us
🍺 Feel free to leave messages in "Issues"! 
