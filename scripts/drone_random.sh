#!/core/bash

ROOTPATH=$(dirname "$PWD")

if [ ! -d "$ROOTPATH/results" ];then
  mkdir -p $ROOTPATH/results/visdrone
fi

PYTHONPATH=$ROOTPATH python3 "$ROOTPATH/analysis/random_rehearsal.py" \
    --random_ratio "0.05"  --cache_max "50" \
    --dataset_name "visdrone"  --dataset_root "$ROOTPATH/datasets/visdrone" \
    --sequences "uav0000124_00944_v" "uav0000073_00600_v" \
    "uav0000150_02310_v" "uav0000342_04692_v" "uav0000218_00001_v"\
    --model "fasterrcnn_mobilenet_v3_large_fpn"  --num_classes 12 \
    --epoch_num 10  --result_dir "$ROOTPATH/results"  --device "cuda:3"


PYTHONPATH=$ROOTPATH python3 "$ROOTPATH/analysis/random_rehearsal.py" \
    --random_ratio "0.10"  --cache_max "100" \
    --dataset_name "visdrone"  --dataset_root "$ROOTPATH/datasets/visdrone" \
    --sequences "uav0000124_00944_v" "uav0000073_00600_v" \
    "uav0000150_02310_v" "uav0000342_04692_v" "uav0000218_00001_v"\
    --model "fasterrcnn_mobilenet_v3_large_fpn"  --num_classes 12 \
    --epoch_num 10  --result_dir "$ROOTPATH/results"  --device "cuda:3"


PYTHONPATH=$ROOTPATH python3 "$ROOTPATH/analysis/random_rehearsal.py" \
    --random_ratio "0.20"  --cache_max "200" \
    --dataset_name "visdrone"  --dataset_root "$ROOTPATH/datasets/visdrone" \
    --sequences "uav0000124_00944_v" "uav0000073_00600_v" \
    "uav0000150_02310_v" "uav0000342_04692_v" "uav0000218_00001_v"\
    --model "fasterrcnn_mobilenet_v3_large_fpn"  --num_classes 12 \
    --epoch_num 10  --result_dir "$ROOTPATH/results"  --device "cuda:3"


PYTHONPATH=$ROOTPATH python3 "$ROOTPATH/analysis/random_rehearsal.py" \
    --random_ratio "0.30"  --cache_max "300" \
    --dataset_name "visdrone"  --dataset_root "$ROOTPATH/datasets/visdrone" \
    --sequences "uav0000124_00944_v" "uav0000073_00600_v" \
    "uav0000150_02310_v" "uav0000342_04692_v" "uav0000218_00001_v"\
    --model "fasterrcnn_mobilenet_v3_large_fpn"  --num_classes 12 \
    --epoch_num 10  --result_dir "$ROOTPATH/results"  --device "cuda:3"