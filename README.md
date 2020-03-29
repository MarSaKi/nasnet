# NasNet 2018

This code is the reimplementation of "Learning Transferable Architectures for Scalable Image Recognition",
including the training process of controller. This code contains three algorithms to search model, Random Search, 
Policy Gradient and PPO. 

## Requirements
```
Python >= 3.6.7, PyTorch == 0.4.0
```  

## Architecture Search
```
python train_search.py --cutout --algorithm RS  #use random search
python train_search.py --cutout --algorithm PG  #use policy gradient
python train_search.py --cutout --algorithm PPO #use PPO
```
Note the validation performance in this step does not indicate the final performance of the architecture. One must train the obtained genotype/architecture from scratch using full-sized models.
Also the default setting is train with 20 processes and 3 GPU. If you want to change the processes to 10, you can use:
```
python train_search.py --cutout --episodes 10
```

## Architecture Evaluation
Because of the limitation of time and computation source, I didn't train the candidate models from scratch. To evaluate architecture, you can train the candidate models obtained from 'Architecture Search' from scratch.

## Results
```
python draw.py
```