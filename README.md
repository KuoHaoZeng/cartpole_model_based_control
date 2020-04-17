## Model-Based Cartpole Control

By Kuo-Hao Zeng, Pengcheng Chen, Mengying Leng, Xiaojuan Wang

### **Proposal**

We are going to learn a CNN or RNN as a dynamic model of the cartpole. This dynamics model aims to predict the current state of the cartpole given the previous observations. We will try different settings such as various combinations of the observations (e.g., image only or state only or image + state), number of observations (e.g., {1, 2, 3} images or states), different model architectures (e.g., CNN, LSTM, GRU) etc. Then, by this learned dynamics model, we are going to learn a NN as a state estimator. The policy further outputs the action based on the state estimation. In this way, we optimize the state estimator by letting the policy mimic the swing up policy via behavior cloning. Furthermore, we have an optional idea about training the dynamic model by some Bayesian NN techniques such as “dynamic model(s)” or “dropout during the inference phase”. It allows us to measure the uncertainty of the prediction produced by the dynamic model(s). However, we leave it as an optional topic since we are not sure if we can make it ontime.

### Set Up

1. Clone this repository

   ```
   git clone git@github.com:KuoHaoZeng/cartpole_model_based_control.git
   ```
   
4. Using `python 3.6`, create a `venv`

   **Note**: The `python` version needs to be above `3.6` to match the original carpole codebase provided by ETAs
   
   ```
   # Create venv and execute it
   python -m venv venv && source venv/bin/activate
   ```
   
4. Install the requirements with

   ```
   # Make sure you execute this under (venv) environment
   pip install -r requirements.txt
   ```

### Environment/Dataset

Cartpole

training data/validation data/testing data

#### Generate your own data

```
TBD
```

### Train and evaluate it!

**Note**: You can always change or adjust the hyperparameters defined in the config file to change the setting such as how often you want to store a checkpoint, how large the learning rate you are going to use, what batch size you are going to use etc.

#### Test the pretrained model on validation/testing set

```
# Download the pretrained model
wget xxx
unzip pretrained.zip && mkdir results && mv pretrained/* results/ && rm pretrained.zip

# Test the model
python main.py --config configs/pretrained_action_sampler_test.yaml
```

#### Train a dynamic model

```
# Train
python main.py --config configs/dm_train.yaml

# Validate or Test
python main.py --config configs/dm_val.yaml
python main.py --config configs/dm_test.yaml
```

#### Train a new state estimator with a trained dynamic model

```
# Prepare the trained forecaster
cd results
mkdir state_esti
mkdir state_esti/checkpoints
cp -r dm/checkpoints/$FORECASTER_YOU_LIKE_TO_USE se/checkpoints/0000000
cd ..

# Train
python main.py --config configs/se_train.yaml

# Validate or Test
python main.py --config configs/se_val.yaml
python main.py --config configs/se_test.yaml
```

#### Main Results for dynamics model

| Model  | L2 difference |
| :-------------: | :-------------: |
| CNN | - |
| LSTM | - |
| GRU | - |

#### Main Results for policy learning

a plot for rewards vs. iterations