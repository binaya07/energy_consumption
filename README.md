# Vehicle Energy Consumption
Deep learning model to predict vehicle energy consumption in various road segments of a city.

## Get started

1. Install dependencies
`pip install -r requirements.txt`

2. Change `config_fig.yaml`'s data_path and model_path variables accordingly to reflect correct PC path

3. Check `model_name` in `config_fig.yaml` for the model to run. And also change `use_lookup` and `use_vehicle_info` parameters according to the model.

4. Run `python train_new.py`

### For example: 

### To run RNN:
```
model_name: RNN
use_lookup: False
use_vehicle_info: False
```
### To run RESNET_BILSTN:
```
model_name: RESNET_BILSTM
use_lookup: True
use_vehicle_info: True
```