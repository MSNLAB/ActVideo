#!/core/bash

ROOTPATH=$(dirname "$PWD")

if [ ! -d "$ROOTPATH/results" ];then
  mkdir -p $ROOTPATH/results/traffic
fi

PYTHONPATH=$ROOTPATH python3 "$ROOTPATH/analysis/active_rehearsal.py" \
    --al_strategy "kmeans"  --al_ratio "0.05"  --cache_max "50" \
    --dataset_name "traffic"  --dataset_root "$ROOTPATH/datasets/traffic" \
    --sequences "traffic_1" "traffic_2" "traffic_3"  \
                "traffic_4" "traffic_5" "traffic_6"  \
    --model "fasterrcnn_mobilenet_v3_large_fpn"  --num_classes 92 \
    --epoch_num 5  --result_dir "$ROOTPATH/results"  --device "cuda:1"


PYTHONPATH=$ROOTPATH python3 "$ROOTPATH/analysis/active_rehearsal.py" \
    --al_strategy "kmeans"  --al_ratio "0.10"  --cache_max "100" \
    --dataset_name "traffic"  --dataset_root "$ROOTPATH/datasets/traffic" \
    --sequences "traffic_1" "traffic_2" "traffic_3"  \
                "traffic_4" "traffic_5" "traffic_6"  \
    --model "fasterrcnn_mobilenet_v3_large_fpn"  --num_classes 92 \
    --epoch_num 5  --result_dir "$ROOTPATH/results"  --device "cuda:1"


PYTHONPATH=$ROOTPATH python3 "$ROOTPATH/analysis/active_rehearsal.py" \
    --al_strategy "kmeans"  --al_ratio "0.20"  --cache_max "200" \
    --dataset_name "traffic"  --dataset_root "$ROOTPATH/datasets/traffic" \
    --sequences "traffic_1" "traffic_2" "traffic_3"  \
                "traffic_4" "traffic_5" "traffic_6"  \
    --model "fasterrcnn_mobilenet_v3_large_fpn"  --num_classes 92 \
    --epoch_num 5  --result_dir "$ROOTPATH/results"  --device "cuda:1"


PYTHONPATH=$ROOTPATH python3 "$ROOTPATH/analysis/active_rehearsal.py" \
    --al_strategy "kmeans"  --al_ratio "0.30"  --cache_max "300" \
    --dataset_name "traffic"  --dataset_root "$ROOTPATH/datasets/traffic" \
    --sequences "traffic_1" "traffic_2" "traffic_3"  \
                "traffic_4" "traffic_5" "traffic_6"  \
    --model "fasterrcnn_mobilenet_v3_large_fpn"  --num_classes 92 \
    --epoch_num 5  --result_dir "$ROOTPATH/results"  --device "cuda:1"