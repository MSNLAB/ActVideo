# Active Continuous Video Analytics

This project explores building an efficient edge video analytics platform
by *Active Continuous Learning* to reduce **human labeling** and **edge
storage**. It helps to judiciously select small but valuable partitions
of video frames for continuous learning in video analytics.

> 🧑‍💻 This project is open for discussion and cooperation,
> please contact us if you are interested.

## Repository Organization

```
├── core/                         # ⚡ Core Python package
│   ├── datasets/                 #    - Dataset loaders for video analytic tasks
│   ├── examplers/                #    - Exampler strategies for rehearsal-based learning
│   ├── metrics/                  #    - Training and testing tools for monitoring
│   ├── models/                   #    - Models for video analytic tasks
│   ├── retrainer/                #    - Continuous learning strategies and interfaces
│   ├── strategies/               #    - Sampling strategies for active learner
│   └── utils/                    #    - Utility functions and classes
│
├── exps/                         # 🔬 Examples of active continuous learning
│   ├── common.py                 #    - Interfaces for common experiment supports
│   ├── core50_nic_exp.py         #    - Experiment on CORe50 benchmark
│   └── video_detection_exp.py    #    - Experiment on video object detections
│
├── scripts/                      # 🛠️ Experiment scripts for running and testing
│
└── videos/                       # 📦 Prepared video datasets for testing
```

## Installation

1. Install all dependencies from `requirements.txt` after creating your
   own python environment with `python >= 3.8.0`.

```shell
$ pip3 install -u -r requirements.txt
```

2. Download prepared datasets (release sooner), and then unzip on the projects root path.

```shell
$ tar -zxvf ***.tar.gz
```

## Examples

We provide working examples for running:

- Classification on [CORe50](https://vlomonaco.github.io/core50/) datasets.

```shell
$ bash ./scripts/core50_***.sh
```

- Object Detecting on video datasets.
    - Dash Camera:  `$ bash ./scripts/dashcam_***.sh`
    - Drone Video:  `$ bash ./scripts/drone_***.sh`
    - Traffic Video:  `$ bash ./scripts/traffic_***.sh`

## Contributing

The codes are expanded on the Active-Learning-as-a-Service
([ALaaS](https://github.com/MLSysOps/Active-Learning-as-a-Service)), and
Video-Platform-as-a-Service ([VPaaS](https://arxiv.org/abs/2102.03012)).

This project is still under demo developing. Pull requests are more than welcome! If you have any questions please feel
free to contact us.

Contact:
[Guanyu Gao](mailto:gygao@njust.edu.cn),
[Lei Zhang](mailto:lei.zhang@njust.edu.cn),
[Huaizheng Zhang](mailto:huaizhen001@e.ntu.edu.sg)

## License
The project is available as open source under the terms of the [Apache 2.0 License](http://www.apache.org/licenses/LICENSE-2.0).