nohup python -u train_search.py






nohup python -u train_search.py --cutout > train_PPO.log 2>&1 &

nohup python -u train_search.py --cutout --algorithm RS > train_RS.log 2>&1 &

nohup python -u train_search.py --cutout --algorithm PG --arch_lr 1e-3 > train_PG.log 2>&1 &