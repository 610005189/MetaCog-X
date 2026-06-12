@echo off
cd /d d:\Projects\MetaCog-X
python -u training\ab_trainer.py --variant metacog --steps 300 --eval_every 50 --print_every 10 --save_csv runs\metacog_300.csv --save_ckpt runs\metacog_300.pt > runs\metacog_300.log 2>&1
