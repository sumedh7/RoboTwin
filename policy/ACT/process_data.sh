task_name=${1}
task_config=${2}
expert_data_num=${3}
num_workers=${4:-1}

python process_data.py $task_name $task_config $expert_data_num --num_workers $num_workers