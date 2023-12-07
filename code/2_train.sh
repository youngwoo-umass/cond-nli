# -------- Environment -----
export PYTHONPATH=src 
model_root=output/model/
data_root=output/tfrecord

# -------- Names ----
run_name=pat
data_name=nli_concat
py_file=mat
job_id=-1

# --------- Train Parameters ---------
step=12500
config_content="{\"batch_size\": 32,
		\"steps_per_execution\": 100,
		\"save_every_n_step\": 5000,
		\"train_step\": ${step},
		\"learning_rate\": 1e-05,
		\"steps_per_epoch\": 12500,
		\"eval_every_n_step\"12500:
}"
config_path=data/${run_name}
echo $config_content > $config_path

output_dir=output/model/${run_name}

log_path=output/log/${run_name}.txt
train_file=${data_root}/${data_name}/train
eval_file=${data_root}/${data_name}/dev
init_checkpoint=${model_root}/uncased_L-12_H-768_A-12/bert_model.ckpt

python3 -u src/run_train.py \
    --config_path=$config_path \
    --init_checkpoint=$init_checkpoint \
    --input_files=$train_file \
    --eval_input_files=$eval_file \
    --run_name=$run_name \
    --action=train \
    --job_id=$job_id \
    --output_dir=$output_dir 

if [ "$?" -ne "0" ];then
       echo "training failed"	
       exit 1
fi

# --------- Eval Parameters ---------
config_content="{\"batch_size\": 16,
		\"steps_per_execution\":1000
}"
echo $config_content > $config_path
output_dir=${model_root}/${run_name}/model_$step
run_name=${run_name}
python3 -u src/run_train.py \
    --config_path=$config_path \
    --input_files=$train_file \
    --eval_input_files=$eval_file \
    --run_name=$run_name \
    --action=eval \
    --job_id=$job_id \
    --output_dir=$output_dir

if [ "$?" -ne "0" ];then
       echo "eval failed"	
       exit 1
fi



