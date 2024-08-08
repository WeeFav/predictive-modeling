
# Predictive Modeling

This repository contains code that predict outcomes based on historical data.

### Structure:
```
predictive-modeling
├── ckpt
├── config
├── datasets
├── model
└── src
```
### Rules:
- remove any column related to ID
- cells that doesn't have any values should be empty
- columns related to numbers must only contain numbers
- if output is date, transform to offset in days

### How to use:    
1. make sure dataset follow the rules above
2. place dataset into /datasets
3. modify /config/preprocessing.py according to dataset filepath, desired input columns, output columns, and any date columns (encoding currently only support 'ordinal' and 'target')
4. modify any of the algorithm yaml files to include hyperparameters
5. run src/full.py
6. model will be saved at model folder
7. run src/predict.py to use model again

### Available algorithms
                    
Regression | Multiclass classification | Binary classification
------------- | -------------- | -----
linear/poly regression  | knn | logistic regression
decision trees  | decision trees | decision trees
random forests  | random forests | random forests
gradient boosting  | gradient boosting | gradient boosting
neural network  | neural network | neural network

### Results and comparison
Result from this repo
Both GTL_STAGED_ISO_ORDER_STATUS_202405270834 and 

GTL_WW_SHIPMENT_REPORT_ARCHIVE_202405271232 take too long to run

-- | Data_for_SAB104_ML_Simulation | seattle-weather | bodyPerformance | car_price_prediction | flight_price | healthcare-dataset-stroke-data | heart_attack | heart_failure | GTL_STAGED_ISO_ORDER_STATUS_202405270834 |GTL_WW_SHIPMENT_REPORT_ARCHIVE_202405271232
-- | - | - | - | - | - | - | - | - | - | -
best model | neural network	| gradient boosting	| gradient boosting	| gradient boosting	| random forest	| logistic regression	| logistic regression	| neural network
performance | r2: 0.998 |	accuracy: 0.84 | accuracy: 0.734	| r2: 0.623	| r2: 0.987 |	accuracy: 0.951 |	accuracy: 0.852 |	accuracy: 0.836
total time taken | < 6 min |	< 1 min |	< 3 min |	< 2 min |	< 26 min |	< 1 min |	< 1 min |	< 1 min

Result from Power Automate

GTL_WW_SHIPMENT_REPORT_ARCHIVE_202405271232 cannot predict OFFSET

-- | Data_for_SAB104_ML_Simulation | seattle-weather | bodyPerformance | car_price_prediction | flight_price | healthcare-dataset-stroke-data | heart_attack | heart_failure | GTL_STAGED_ISO_ORDER_STATUS_202405270834 |GTL_WW_SHIPMENT_REPORT_ARCHIVE_202405271232
-- | - | - | - | - | - | - | - | - | - | - 
performance | r2: 0.97 |	accuracy: 0.85 |	accuracy: 0.77 |	r2: 0.287 |	r2: 0.74 |	accuracy: 0.81 |	accuracy: 0.90 |	accuracy: 0.78 |	r2: 0.97
total time taken | < 3 min |	< 3 min |	< 3 min |	< 3 min |	< 3 min |	< 3 min |	< 3 min |	< 3 min | < 3 min

### Dataset
[Link](https://keysighttech.sharepoint.com/:f:/r/sites/ImageSense/Shared%20Documents/Predictive%20Modeling%20Dataset?csf=1&web=1&e=ihaRU0)


### Installation
```bash
  pip install -r requirements.txt
```

### TODO
- automatically detect hyperparameters
- add support for other encoding methods






