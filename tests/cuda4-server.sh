pkill -f 'tests/cifar10-resnet9.py'
mkdir $1
DMLC_NUM_WORKER=2 DMLC_NUM_SERVER=1 DMLC_PS_ROOT_URI=128.122.49.167 DMLC_PS_ROOT_PORT=9800 CUDA_VISIBLE_DEVICES='' DMLC_ROLE=server python -u ../tests/cifar10-resnet9.py --log $1 > $1/server.log 2>$1/server.err