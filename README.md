# Age Prediction through Hyperbolic Radius Extraction from Hyperbolic Embeddings of MEG Functional Connectivity Brain Networks using FHNN 
Age Prediction research findings available in bioRxiv here: https://www.biorxiv.org/content/10.1101/2024.10.01.616153v1. Based on [Fully Hyperbolic Neural Networks](https://arxiv.org/abs/2105.14686) 

```
@article{chen2021fully,
  title={Fully Hyperbolic Neural Networks},
  author={Chen, Weize and Han, Xu and Lin, Yankai and Zhao, Hexu and Liu, Zhiyuan and Li, Peng and Sun, Maosong and Zhou, Jie},
  journal={arXiv preprint arXiv:2105.14686},
  year={2021}
}
{"mode":"full","isActive":false}
```

# FHNN and Age Prediction Source Code:
Source for FHNN code based on [HGCN](https://github.com/HazyResearch/hgcn) and [FHNN](https://github.com/chenweize1998/fully-hyperbolic-nn) repositories. Age Prediction code is entirely own.

```
📦gcn
 ┣ age_prediction.py    # Regression Models to Predict Age  
 ┣ age_predictor_utils.py
 ┣ 📂data
 ┣ 📂layers
 ┃ ┣ 📜__init__.py
 ┃ ┣ 📜att_layers.py
 ┃ ┣ 📜hyp_layers.py    # Defines Lorentz Graph Convolutional Layer
 ┃ ┗ 📜layers.py
 ┣ 📂manifolds
 ┃ ┣ 📜__init__.py
 ┃ ┣ 📜base.py
 ┃ ┣ 📜euclidean.py
 ┃ ┣ 📜hyperboloid.py
 ┃ ┣ 📜lmath.py         # Math related to our manifold
 ┃ ┣ 📜lorentz.py       # Our manifold
 ┃ ┣ 📜poincare.py
 ┃ ┗ 📜utils.py
 ┣ 📂models
 ┃ ┣ 📜__init__.py
 ┃ ┣ 📜base_models.py
 ┃ ┣ 📜decoders.py      # Include FHNN decoder
 ┃ ┗ 📜encoders.py      # Include FHNN encoder
 ┣ 📂optim
 ┣ 📂utils
 ┣ 📂logs               # (Generated once FHNN model is ran i.e. hyperbolic embeddings have been generated)
 ┃ ┣ 📂lp
 ┃ ┣ ┣ 📂{date}         # `date` is the date of execution of the hyperbolic embedding generation, in `YYYY-MM-DD` format, 
 ┃ ┣ ┣ ┣ 📂{run_number} # `run_number` corresponds to the number of times the `train_graph_iteration.py` has been executed during that particular day, 0-indexed
 ┃ ┣ ┣ ┣ ┣ 📂embeddings
 ┃ ┣ ┣ ┣ ┣ 📜log.txt
 ┣ train_graph_iteration.py   # Main file for hyperbolic embedding generation
 ```

## Workflows:

## 1. Hyperbolic Embedding Generation (Training FHNN model on Link Prediction of MEG Functional Connectivity using Graph Iteration)
To generate hyperbolic embeddings from the Cam-CAN dataset, open a Jupyter notebook and run the following:


```
! python train_graph_iteration.py \
    --task lp \
    --act None \
    --dataset cam_can_multiple\
    --model HyboNet \
    --lr 0.05 \
    --dim 3 \
    --num-layers 2 \
    --bias 1 \
    --dropout 0.25 \
    --weight-decay 1e-3 \
    --manifold Lorentz \
    --log-freq 5 \
    --cuda -1 \
    --patience 500 \
    --grad-clip 0.1 \
    --seed 1234 \
    --save 1
```

This will produce hyperbolic embeddings, which will be saved in the `/gcn/logs/lp/{date}/{run_number}/embeddings` subdirectory, where `date` is the date of execution of the hyperbolic embedding generation, in `YYYY-MM-DD` format, and `run_number` corresponds to the number of times the `train_graph_iteration.py` has been executed during that particular day, 0-indexed. 

For example, a hyperbolic embedding generation run on the Cam-CAN dataset performed on August, 12th, 2024 after 2 previous runs will output the generated embeddings in `/gcn/logs/lp/2024_8_12/2/embeddings/embeddings_{data_split}_{subject_number}` as Numpy arrays. Say subject 7 was assigned to the test split, then if we would like to look at the hyperbolic embeddings for subject 7 in the test data split, we would look into `/gcn/logs/lp/2024_8_12/2/embeddings/embeddings_test_7.npy`. An initial age prediction regression will be performed using these embeddings, and the results will be outputted within the logs of the run in `/gcn/logs/lp/{date}/{run_number}/log.txt` (so in our example the initial regression results would be in `/gcn/logs/lp/2024_8_12/2/log.txt`). To perform additional subsequent regressions, see Workflow # 2 Age Prediction.

Arguments passed to program:

`--task` Specifies task. Should be [lp], lp denotes link prediction.

`--dataset` Specifies dataset.

`--lr` Specifies learning rate.

`--dim` Specifies dimension of embeddings.

`--num-layers` Specifies number of layers.

`--bias` To enable the bias, set it to 1.

`--dropout` Specifies dropout rate.

`--weight-decay` Specifies weight decay value.

`--log-freq` Interval for logging.

For other arguments, see `config.py`

## 2. Age Prediction using Generated Hyperbolic Embeddings

Choose an existing hyperbolic embedding outputs directory in `/gcn/logs/lp/{date}/{run_number}` (note that you will need to have run Workflow #1 first). Then, open a Jupyter notebook and run the following to perform an age prediction regression based on the hyperbolic radius of the hyperbolic embeddings!  

```
from age_predictor import Age_Predictor 
regression_type_str = "ridge"       # Other choices supported: 'random_forest', 'ada_boost'
date = '2024_8_12'                  # Change accordingly 
run_number = '2'                    # Change accordingly

age_predictor_model = Age_Predictor(
        date, 
        run_number, 
        regression_type_str)
age_predictor_model.regression()
```
