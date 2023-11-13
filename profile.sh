NCU_ARGS="--metrics dram__bytes_read,gpu__time_duration --clock-control none --target-processes all"
NAME="tvm_meta_schedule_resnet-18"

export PYTHONPATH=/home2/xiachunwei/Software/tvm/python
export LD_LIBRARY_PATH=/home2/xiachunwei/Software/tvm/build:/usr/local/cuda-11.7/lib64:/usr/local/cuda-11.7/targets/x86_64-linux/lib
sudo PYTHONPATH=${PYTHONPATH} LD_LIBRARY_PATH=${LD_LIBRARY_PATH} /usr/local/cuda-11.7/bin/ncu ${NCU_ARGS} -o ncu-${NAME} -f /home2/xiachunwei/anaconda3/bin/python3 relay_tune_resnet-18.py --run
ncu -i ./ncu-${NAME}.ncu-rep --csv --page raw  > ncu-${NAME}.csv
python3 extract_ncu_cuda_kernel_latency.py ncu-${NAME}.csv
