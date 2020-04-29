pkill -f 'tests/mnist.py' || true
export DMLC_NUM_WORKER=4 DMLC_NUM_SERVER=1 DMLC_PS_ROOT_URI=127.0.0.1 DMLC_PS_ROOT_PORT=9800
echo "using $DMLC_PS_ROOT_URI:$DMLC_PS_ROOT_PORT"

# sleep 10
DMLC_ROLE=worker DMLC_WORKER_ID=0 WORKER_IP=127.0.0.1 python -u ../tests/mnist.py > worker0.log 2>worker0.err || true &
# sleep 5
DMLC_ROLE=worker DMLC_WORKER_ID=1 WORKER_IP=127.0.0.1 python -u ../tests/mnist.py > worker1.log 2>worker1.err || true &
DMLC_ROLE=worker DMLC_WORKER_ID=2 WORKER_IP=127.0.0.1 python -u ../tests/mnist.py > worker2.log 2>worker2.err || true &
DMLC_ROLE=worker DMLC_WORKER_ID=3 WORKER_IP=127.0.0.1 python -u ../tests/mnist.py > worker3.log 2>worker3.err || true &
CUDA_VISIBLE_DEVICES='' DMLC_ROLE=server python -u ../tests/mnist.py > server.log 2>server.err || true

# read 

# pkill -SIGINT -f 'tests/mnist.py' || true

# sleep 3
# pkill -f 'tests/mnist.py' || true
