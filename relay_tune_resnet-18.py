import os
import argparse
import numpy as np

import tvm
from tvm import relay
from tvm.relay import testing
from tvm import meta_schedule as ms
from tvm.target import Target
from tvm.contrib import graph_executor


def get_network(name, batch_size, dtype="float32"):
    """Get the symbol definition and random weight of a network"""
    input_shape = (batch_size, 3, 224, 224)
    output_shape = (batch_size, 1000)

    if "resnet" in name:
        n_layer = int(name.split("-")[1])
        mod, params = testing.resnet.get_workload(
            num_layers=n_layer, batch_size=batch_size, dtype=dtype
        )
    elif "vgg" in name:
        n_layer = int(name.split("-")[1])
        mod, params = testing.vgg.get_workload(
            num_layers=n_layer, batch_size=batch_size, dtype=dtype
        )
    elif name == "mobilenet":
        mod, params = testing.mobilenet.get_workload(batch_size=batch_size, dtype=dtype)
    elif name == "squeezenet_v1.1":
        mod, params = testing.squeezenet.get_workload(
            batch_size=batch_size, version="1.1", dtype=dtype
        )
    elif name == "inception_v3":
        input_shape = (batch_size, 3, 299, 299)
        mod, params = testing.inception_v3.get_workload(batch_size=batch_size, dtype=dtype)
    elif name == "mxnet":
        # an example for mxnet model
        from mxnet.gluon.model_zoo.vision import get_model

        block = get_model("resnet18_v1", pretrained=True)
        mod, params = relay.frontend.from_mxnet(block, shape={input_name: input_shape}, dtype=dtype)
        net = mod["main"]
        net = relay.Function(
            net.params, relay.nn.softmax(net.body), None, net.type_params, net.attrs
        )
        mod = tvm.IRModule.from_expr(net)
    else:
        raise ValueError("Unsupported network: " + name)

    return mod, params, input_shape, output_shape


def tune_resnet_18():
    mod, params, input_shape, output_shape = get_network("resnet-18", 1)
    target = Target("nvidia/nvidia-a100")
    work_dir = "resnet_18_meta_scheduler"
    database = ms.relay_integration.tune_relay(mod, params, target, work_dir, 
        max_trials_global=30*1000, max_trials_per_task=1000, num_tuning_cores="logical")
    runtime_module = ms.relay_integration.compile_relay(database, mod, target, params)
    runtime_module.export_library("resnet_18.so")


def compile_database_to_library():
    mod, params, input_shape, output_shape = get_network("resnet-18", 1)
    target = Target("nvidia/nvidia-a100")
    work_dir = "resnet_18_meta_scheduler"
    database = ms.database.JSONDatabase(work_dir=work_dir)
    runtime_module = ms.relay_integration.compile_relay(database, mod, target, params)
    runtime_module.export_library("resnet_18.so")


def inspect_tuning_records():
    work_dir = "resnet_18_meta_scheduler_sample_2"
    database = ms.database.JSONDatabase(work_dir=work_dir)
    all_tuning_records = database.get_all_tuning_records()
    workload_tuning_record_map = {}
    for record in all_tuning_records:
        if record.workload in workload_tuning_record_map.keys():
            workload_tuning_record_map[record.workload].append(record)
        else:
            workload_tuning_record_map[record.workload] = [record]
    
    for workload, record_list in workload_tuning_record_map.items():
        print(workload.mod)
        for record in record_list:
            print(record.run_secs)
        print("**"*10)


def run_resnet_18():
    loaded_lib = tvm.runtime.load_module("resnet_18.so")
    img = np.ones((1, 3, 224, 224), dtype=np.float32)
    input_data = tvm.nd.array(img)
    dev = tvm.cuda()
    module = graph_executor.GraphModule(loaded_lib["default"](dev))
    module.run(data=input_data)
    out_deploy = module.get_output(0).numpy()
    # Print first 10 elements of output
    print(out_deploy.flatten()[0:10])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='relay_tune_resnet-18',
                    description='Using tvm meta schedule to tune and run resnet-18',
                    epilog='Text at the bottom of help')
    parser.add_argument('--tune', action='store_true', help='tune resnext-18 and save to module')
    parser.add_argument('--run', action='store_true', help='load module and run resnext-18')
    args = parser.parse_args()
    if args.tune:
        tune_resnet_18()
    elif args.run:
        run_resnet_18()
    else:
        print("Please tune of run this program")
