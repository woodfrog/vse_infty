DATASET_NAME='coco'
DATA_PATH='data/'${DATASET_NAME}
VOCAB_PATH='data/vocab'
WEIGHT_PATH='data/weights'

# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train.py \
CUDA_VISIBLE_DEVICES=4,5,6,7 python3 train.py \
  --data_path ${DATA_PATH} --data_name ${DATASET_NAME} --vocab_path ${VOCAB_PATH} \
  --logger_name runs/${DATASET_NAME}_vsepp_updown_bert_var_gpool_test/log --model_name runs/${DATASET_NAME}_vsepp_updown_bert_var_gpool_test \
  --cross_attn=none --num_epochs=25 --lr_update=15 --learning_rate=5e-4 --precomp_enc_type backbone  --workers 20 --backbone_source vsepp_detector \
  --vse_mean_warmup_epochs 1 --backbone_warmup_epochs 0 --embedding_warmup_epochs 1  --optim adam --backbone_lr_factor 0.01  --log_step 200 \
  --input_scale_factor 2.0  --backbone_path ${WEIGHT_PATH}/original_updown_backbone.pth  --drop
