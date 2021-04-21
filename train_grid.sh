DATASET_NAME='coco'
DATA_PATH='/tmp/data/'${DATASET_NAME}
WEIGHT_PATH='/tmp/data/weights'
VOCAB_PATH='/tmp/data/vocab'


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 train.py \
  --data_path ${DATA_PATH} --data_name ${DATASET_NAME}  --vocab_path ${VOCAB_PATH}\
  --logger_name runs/${DATASET_NAME}_butd_grid_bigru_gpo/log --model_name runs/${DATASET_NAME}_butd_grid_bigru \
  --num_epochs=25 --lr_update=15 --learning_rate=5e-4 --precomp_enc_type backbone  --workers 20 --backbone_source detector \
  --vse_mean_warmup_epochs 1 --backbone_warmup_epochs 0 --embedding_warmup_epochs 1  --optim adam --backbone_lr_factor 0.01  --log_step 200 \
  --input_scale_factor 2.0  --backbone_path ${WEIGHT_PATH}/original_updown_backbone.pth
