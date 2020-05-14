pkill -f 'tests/cifar10-resnet9.py' || true
export DMLC_NUM_WORKER=2 DMLC_NUM_SERVER=1 DMLC_PS_ROOT_URI=128.122.49.167 DMLC_PS_ROOT_PORT=9800
echo "using $DMLC_PS_ROOT_URI:$DMLC_PS_ROOT_PORT"
mkdir $1

# ssh cuda2 "cd /scratch/jl10439/network-project/build && bash ../tests/cuda24-server.sh $1 " || true &
# sleep 10
DMLC_ROLE=worker DMLC_WORKER_ID=0 WORKER_IP=128.122.49.166 python -u ../tests/cifar10-resnet9.py --log $1 > $1/worker0.log 2>$1/worker0.err || true &
# sleep 5
CUDA_VISIBLE_DEVICES='1,0' DMLC_ROLE=worker DMLC_WORKER_ID=1 WORKER_IP=128.122.49.166 python -u ../tests/cifar10-resnet9.py --log $1 > $1/worker1.log 2>$1/worker1.err || true
# DMLC_ROLE=worker DMLC_WORKER_ID=2 WORKER_IP=128.122.49.166 python -u ../tests/cifar10-resnet9.py --log $1 > $1/worker2.log 2>$1/worker2.err || true &
# DMLC_ROLE=worker DMLC_WORKER_ID=3 WORKER_IP=128.122.49.166 python -u ../tests/cifar10-resnet9.py --log $1 > $1/worker3.log 2>$1/worker3.err || true &

# read 

# pkill -SIGINT -f 'tests/cifar10-resnet9.py' || true

# sleep 3
# pkill -f 'tests/cifar10-resnet9.py' || true
