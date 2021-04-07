DATASET_NAME='coco'
DATA_PATH='data/'${DATASET_NAME}

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train.py \
  --data_path ${DATA_PATH} --data_name ${DATASET_NAME} \
  --logger_name runs/${DATASET_NAME}_entity_bert_gpo_test/log --model_name runs/${DATASET_NAME}_entity_bert_gpo_test \
  --cross_attn=none --num_epochs=25 --lr_update=15 --learning_rate=.0005 --precomp_enc_type basic --workers 10 \
  --log_step 200 --embed_size 1024 --vse_mean_warmup_epochs 1
