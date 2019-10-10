export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=5

log_name=log3
nohup python3 -u main.py > $log_name 2>&1 &
tail -f $log_name 
