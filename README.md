# Codebase for "ADRL: An Offline Reinforcement Learning Method based on Anomaly Detection Mechanism for Recommending Safe Treatment"


This directory contains implementations of ADRL framework for offline reinforcement learning using liver cancer ablation EHR dataset.

Simply run python3 -m main.py

### Code explanation

(1) preprocess/load_data.py
- Load data

(2) preprocess/get_dataset.py
- Data preprocessing

(3) preprocess/missing_values_imputation.py
- Simple imputate missing values in dataset

(4) preprocess/representation_learning.py
- Define autoencoder framework to encode static and dynamic variables

(5) model/ae.py
- Define and return an autoencoder model

(6) model/evaluate.py
- Performance of computation in treatment recommendation tasks

(7) model/ReinforcementLearning.py
- Define ADRL framework

(8) model/TD3/
- Classic TD3 Model

(9) main.py
- A framework for data loading, model construction, result evaluation

(10) arguments.py
- Parameter settings

Note that hyper-parameters should be optimized for different datasets.


## Main Dependency Library

scikit-learn==0.24.2

torch==1.8.2

torchaudio==0.8.2

torchvision==0.9.2
