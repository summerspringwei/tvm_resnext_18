
## Using TVM meta schedule to tune and run resnet-18

Requrements:
* tvm==0.15.0dev
* cuda-11.7
* nsight-compute 


Hardware:
This is tuned on NVIDIA-A100 GPU.

How to tune:
```shell
python3 relay_tune_resnet-18.py --tune
```

We have already tune the resnet-18 with the following arguments:
`max_trials_global=30*1000, max_trials_per_task=1000`.

How to run:
```shell
python3 relay_tune_resnet-18.py --run
```

How to profile the end-to-end latency using `ncu`:
```shell
bash profile.sh
```
Note: you may need to change the path to `ncu` and set env `PYTHONPATH` and `LD_LIBRARY_PATH` to the TVM path.