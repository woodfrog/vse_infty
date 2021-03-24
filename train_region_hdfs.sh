DATA_PATH='data/features'
VOCAB_PATH='data/vocab'
DATASET_NAME='coco_precomp'

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train.py \
  --data_path ${DATA_PATH} --data_name ${DATASET_NAME} --vocab_path ${VOCAB_PATH} \
  --logger_name runs/${DATASET_NAME}_entity_bert_var_gpool_local/log --model_name runs/${DATASET_NAME}_entity_bert_var_gpool_local \
  --cross_attn=none --num_epochs=25 --lr_update=15 --learning_rate=.0005 --img_pool gpool --precomp_enc_type basic --workers 10 \
  --log_step 200 --embed_size 1024 --vse_mean_warmup_epochs 1 --drop  --resume runs/${DATASET_NAME}_entity_bert_var_gpool_local/checkpoint.pth.tar
