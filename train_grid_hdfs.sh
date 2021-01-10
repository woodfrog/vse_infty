CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 train.py --data_path hdfs:///home/byte_arnold_hl_vc/user/chenjiacheng/data/coco --data_name coco_precomp --vocab_path hdfs:///home/byte_arnold_hl_vc/user/chenjiacheng/data/vse_infty/vocab --logger_name runs/coco_vsepp_updown_bert_var_gpool_test/log --model_name runs/coco_vsepp_updown_bert_var_gpool_test  --cross_attn=none --num_epochs=25 --lr_update=15 --learning_rate=.0005 --precomp_enc_type backbone  --workers 30 --backbone_source vsepp_detector --vse_mean_warmup_epochs 1 --backbone_warmup_epochs 0 --embedding_warmup_epochs 1  --optim adam --backbone_lr_factor 0.01  --log_step 200 --input_scale_factor 2.0  --bi_gru  --backbone_path hdfs:///home/byte_arnold_hl_vc/user/chenjiacheng/data/coco/original_updown/original_updown_backbone.pth  --drop

#--backbone_path hdfs:///home/byte_arnold_hl_vc/user/chenjiacheng/data/vse_infty/ssl_weights/simclr-resnet101-1x-sk0-sp.pth

#--resume runs/coco_vsepp_updown_learn_mean_large_temp_0.01/checkpoint_1.pth.tar



