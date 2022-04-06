# Installation
Locate at the root of the project in your terminal where you can see `setup.py`, then install the module using `pip`

```bash
 $ pip install .
 ```

# Usage
```bash
$ macorp.forecast [OPTIONS] COMMAND [ARGS]...
```

### Training
Example:
```bash
$ macorp.forecast train-chat-forecast -p data/chat_demand.csv -i
```
A `.pkl` is created under `deployed_models` that contains the trained model which will be used for inference.

A `.png` is created under `deployed_models` that visualizes the performance of the model on a validation subset.  


### Inference for single entry
Example:
```bash
$ macorp.forecast chat-forecast -d deployed_models/<deployment_name>.pkl -t 2017-07-01 -e 35000
```
### Inference for a batch
Example:
```bash
$ macorp.forecast chat-forecast-batch -p deployed_models/<deployment_name>.pkl -d data/chat_demand.csv -i docs/chat_forecast_visual.png
```
The entries are fetched from a csv file stored with the following columns (date, eligible_users) or (date, eligible_users, chats). If there are already entries under chats column, they are excluded from inference but are visualized.

### Nurse recommendation for a single entry
Example:
```bash
$ macorp.forecast staff-recom -c 450 -h 20 -w 10 -r 0.1 -s 0.9
```
In this example, the number of nurses required (for each working shift) of a day that 450 chats are expected is recommended. The expected wait time and time of providing service to each client is considered to be 10 and 20 minutes respectively.
It is also assumed that 0.1 of time of each agent is spent on refreshment and self-care. 

### Nurse recommendation batch mode
Example:
```bash
$ macorp.forecast staff-recom-batch -p deployed_models/LinearRegression___2022-03-31.pkl -d data/chat_demand.csv -h 20 -w 10 -r 0.1 -s 0.9 -i docs/nurse_recom.png
```


---

## Note
- **Explanatory analysis and Q&A is provided in a notebook under `notebooks`.**
- **For documentation of the modules, methods and classes please click on [this link](../docs/macorp.html) to a pydoc generated html file**