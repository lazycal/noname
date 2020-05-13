pkill -f 'tests/cifar10-resnet9.py' || true
export DMLC_NUM_WORKER=2 DMLC_NUM_SERVER=1 DMLC_PS_ROOT_URI=127.0.0.1 DMLC_PS_ROOT_PORT=9800
echo "using $DMLC_PS_ROOT_URI:$DMLC_PS_ROOT_PORT"

CUDA_VISIBLE_DEVICES='' DMLC_ROLE=server python -u ../tests/cifar10-resnet9.py > cifar10-resnet9-lossy/server.log 2>cifar10-resnet9-lossy/server.err || true &
sleep 2
DMLC_ROLE=worker DMLC_WORKER_ID=0 WORKER_IP=127.0.0.1 python -u ../tests/cifar10-resnet9.py > cifar10-resnet9-lossy/worker0.log 2>cifar10-resnet9-lossy/worker0.err || true &
# sleep 5
DMLC_ROLE=worker DMLC_WORKER_ID=1 WORKER_IP=127.0.0.1 python -u ../tests/cifar10-resnet9.py > cifar10-resnet9-lossy/worker1.log 2>cifar10-resnet9-lossy/worker1.err || true
# DMLC_ROLE=worker DMLC_WORKER_ID=2 WORKER_IP=127.0.0.1 python -u ../tests/cifar10-resnet9.py > cifar10-resnet9-lossy/worker2.log 2>cifar10-resnet9-lossy/worker2.err || true &
# DMLC_ROLE=worker DMLC_WORKER_ID=3 WORKER_IP=127.0.0.1 python -u ../tests/cifar10-resnet9.py > cifar10-resnet9-lossy/worker3.log 2>cifar10-resnet9-lossy/worker3.err || true &

# read 

# pkill -SIGINT -f 'tests/cifar10-resnet9.py' || true

# sleep 3
# pkill -f 'tests/cifar10-resnet9.py' || true
