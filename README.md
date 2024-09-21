Thanks for Tsinghua \url{https://github.com/thuml/Time-Series-Library}


run example:
CUDA_VISIBLE_DEVICES=1 nohup python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh2.csv --model_id ETTh2_96_96 --model Fi2VTS --data ETTh2 --features M --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_layers 1 --factor 1 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --d_model 32 --d_ff 32 --top_k 10 --itr 1 &
