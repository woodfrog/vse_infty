CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train.py --data_path hdfs://haruna/home/byte_arnold_hl_vc/user/chenjiacheng/data/f30k --data_name f30k_precomp --vocab_path hdfs://haruna/home/byte_arnold_hl_vc/user/chenjiacheng/data/vse_infty/vocab --logger_name runs/f30k_entity_bert_var_gpool/log --model_name runs/f30k_entity_bert_var_gpool  --bi_gru --cross_attn=none --num_epochs=25 --lr_update=15 --learning_rate=.0005 --img_pool gpool --precomp_enc_type basic --workers 10 --log_step 200 --embed_size 1024 --vse_mean_warmup_epochs 1 --drop


#--resume runs/coco_all_learn_2048_drop_temp_0.01_test/checkpoint_19.pth.tar
