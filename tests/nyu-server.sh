pkill -f 'tests/cifar10-resnet9.py'
mkdir $1
DMLC_NUM_WORKER=2 DMLC_NUM_SERVER=1 DMLC_PS_ROOT_URI=172.24.71.205 DMLC_PS_ROOT_PORT=9800 DMLC_ROLE=server python -u ../tests/cifar10-resnet9.py --no-cuda --log $1 > $1/server.log 2>$1/server.err