#!/usr/bin/env bash

cd ../clients

#python bow_bert_emb_lc_gensect.py \
#--exp_name "debug_bow_bert_emb_gensect" \
#--device cpu \
#--bert_type "bert-large-cased" \
#--max_length 10 \
#--max_num_words 15000 \
#--debug \
#--debug_dataset_proportion 0.01 \
#--bs 32 \
#--emb_dim 1024 \
#--lr 1e-2 \
#--epochs 1 \
#--save_every 1 \
#--log_train_metrics_every 50

python bow_bert_emb_lc_gensect.py \
--exp_name "bow_bert_emb_gensect_base_cased_1e-2lr_10e" \
--device cuda:0 \
--bert_type "bert-base-cased" \
--max_length 10 \
--max_num_words 15000 \
--debug_dataset_proportion 0.01 \
--bs 32 \
--emb_dim 1024 \
--lr 1e-2 \
--epochs 10 \
--save_every 5 \
--log_train_metrics_every 50

python bow_bert_emb_lc_gensect.py \
--exp_name "bow_bert_emb_gensect_large_cased_1e-2lr_10e" \
--device cuda:0 \
--bert_type "bert-large-cased" \
--max_length 10 \
--max_num_words 15000 \
--debug_dataset_proportion 0.01 \
--bs 32 \
--emb_dim 1024 \
--lr 1e-2 \
--epochs 10 \
--save_every 5 \
--log_train_metrics_every 50

python bow_bert_emb_lc_gensect.py \
--exp_name "bow_bert_emb_gensect_large_uncased_1e-2lr_10e" \
--device cuda:0 \
--bert_type "bert-large-uncased" \
--max_length 10 \
--max_num_words 15000 \
--debug_dataset_proportion 0.01 \
--bs 32 \
--emb_dim 1024 \
--lr 1e-2 \
--epochs 10 \
--save_every 5 \
--log_train_metrics_every 50

python bow_bert_emb_lc_gensect.py \
--exp_name "bow_bert_emb_gensect_sci_cased_1e-2lr_10e" \
--device cuda:0 \
--bert_type "scibert-sci-cased" \
--max_length 10 \
--max_num_words 15000 \
--debug_dataset_proportion 0.01 \
--bs 32 \
--emb_dim 768 \
--lr 1e-2 \
--epochs 10 \
--save_every 5 \
--log_train_metrics_every 50

python bow_bert_emb_lc_gensect.py \
--exp_name "bow_bert_emb_gensect_sci_uncased_1e-2lr_10e" \
--device cuda:0 \
--bert_type "scibert-sci-uncased" \
--max_length 10 \
--max_num_words 15000 \
--debug_dataset_proportion 0.01 \
--bs 32 \
--emb_dim 768 \
--lr 1e-2 \
--epochs 20 \
--save_every 5 \
--log_train_metrics_every 50