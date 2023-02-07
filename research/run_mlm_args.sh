python run_mlm.py --model_type albert --tokenizer_name /home/penguin/Albert/TRAIN_ALBERT/albert-tokenizer \
       --train_file /home/penguin/Albert/geo_problem.txt --validation_split_percentage 5 --max_seq_length 256 \
       --preprocessing_num_workers 12 --output_dir albert-output/ --do_train True --do_eval True --do_predict True \
       --evaluation_strategy epoch --per_device_train_batch_size 12 --per_device_eval_batch_size 12 \
       --gradient_accumulation_steps 8 --eval_accumulation_steps 8 --logging_dir /home/penguin/Albert/TRAIN_ALBERT/experiment/log \
       --logging_strategy epoch --save_strategy epoch --save_total_limit 5 --fp16 True
