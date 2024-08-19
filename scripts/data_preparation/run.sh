#!/bin/bash
# spatial-temporal forecasting
python scripts/data_preparation/PEMS04/generate_training_data.py --history_seq_len 12 --future_seq_len 12
python scripts/data_preparation/PEMS08/generate_training_data.py --history_seq_len 12 --future_seq_len 12

# long-term time series forecasting
python scripts/data_preparation/PEMS04/generate_training_data.py --history_seq_len 336 --future_seq_len 336
python scripts/data_preparation/PEMS08/generate_training_data.py --history_seq_len 336 --future_seq_len 336
