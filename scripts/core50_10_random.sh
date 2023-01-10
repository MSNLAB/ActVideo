#!/core/bash

ROOTPATH=$(dirname "$PWD")

if [ ! -d "$ROOTPATH/results" ];then
  mkdir -p $ROOTPATH/results/dashcam
fi

PYTHONPATH=$ROOTPATH python3 "$ROOTPATH/analysis/core50_nic_exp.py" \
    --al_strategy "random"  --al_ratio "0.05"  --cache_max "100" \
    --core50_root "$ROOTPATH/datasets/core50" \
    --core50_cumul "8"  --core50_category "10" \
    --epoch_num "5"  --result_dir "$ROOTPATH/results"  --device "cuda:0"


PYTHONPATH=$ROOTPATH python3 "$ROOTPATH/analysis/core50_nic_exp.py" \
    --al_strategy "random"  --al_ratio "0.10"  --cache_max "200" \
    --core50_root "$ROOTPATH/datasets/core50" \
    --core50_cumul "8"  --core50_category "10" \
    --epoch_num "5"  --result_dir "$ROOTPATH/results"  --device "cuda:0"


PYTHONPATH=$ROOTPATH python3 "$ROOTPATH/analysis/core50_nic_exp.py" \
    --al_strategy "random"  --al_ratio "0.20"  --cache_max "400" \
    --core50_root "$ROOTPATH/datasets/core50" \
    --core50_cumul "8"  --core50_category "10" \
    --epoch_num "5"  --result_dir "$ROOTPATH/results"  --device "cuda:0"


PYTHONPATH=$ROOTPATH python3 "$ROOTPATH/analysis/core50_nic_exp.py" \
    --al_strategy "random"  --al_ratio "0.30"  --cache_max "600" \
    --core50_root "$ROOTPATH/datasets/core50" \
    --core50_cumul "8"  --core50_category "10" \
    --epoch_num "5"  --result_dir "$ROOTPATH/results"  --device "cuda:0"


PYTHONPATH=$ROOTPATH python3 "$ROOTPATH/analysis/core50_nic_exp.py" \
    --al_strategy "random"  --al_ratio "0.99"  --cache_max "2000" \
    --core50_root "$ROOTPATH/datasets/core50" \
    --core50_cumul "8"  --core50_category "10" --num_classes "10" \
    --epoch_num "5"  --result_dir "$ROOTPATH/results"  --device "cuda:0"