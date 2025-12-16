# COCCL_PPoPP_AE

## Build

To build the library :

```shell
git clone https://github.com/maomaohpp/COCCL_PPoPP_AE.git
chmod 777 -R COCCL_PPoPP_AE
cd COCCL_PPoPP_AE
bash build.sh /path/to/cuda \
/path/to/mpi \
/path/to/COCCL_PPoPP_AE \
"-gencode=arch=compute_90,code=sm_90"
# use the corresponding NVCC_GENCODE for your hardware
```

## Intrgrated with Framework

Please install the necessary environment dependencies for training frameworks such as [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) or [PyTorch](https://github.com/pytorch/pytorch).

To switch from the NCCL library to the COCCL library, follow the steps below:

1. Confirm whether the NCCL library used by PyTorch is a dynamic library.

   - Confirm the location of the PyTorch library.
     If you know that PyTorch is installed in a specific directory, you can search directly within that directory. For example, after confirming that PyTorch resides in `/usr/local/lib`, running the query command successfully pinpointed the exact path to the `libtorch.so` file, as shown below:

     ```bash
     find /usr/local/lib -name "libtorch*"
     # The example results are as follows:
     /usr/local/lib/python3.10/dist-packages/torch/lib/libtorchcuda.so
     /usr/local/lib/python3.10/dist-packages/torch/lib/libtorch.so
     /usr/local/lib/python3.10/dist-packages/torch/lib/libtorchbindtest.so
     ```

   - Use the `ldd` command to inspect the PyTorch library’s dependency on the NCCL library.

     ```bash
     ldd libtorch.so | grep nccl
     ```

     If the command returns results in the following format, it indicates that PyTorch depends on NCCL as a dynamic library. You may then proceed to configure COCCL according to the subsequent steps.

     ```bash
     libnccl.so.2=>/usr/lib/x86_64-linux-gnu/libnccl.so.2(0x00007feab3b27000)
     ```

     If the command returns no results, this indicates that PyTorch relies on NCCL as a static (non-dynamic) library and therefore cannot be switched to COCCL. To proceed with COCCL configuration, you must use a PyTorch version that depends on the NCCL dynamic library.

2. Before running the training script, please specify the paths to COCCL and the compressor dynamic libraries in COCCL_PPoPP_AE/training_scripts/training_envs.sh.

   ```bash
   #!/bin/bash
   export CUDA_PATH=/path/to/cuda
   export COCCL_PATH=/path/to/COCCL_PPoPP_AE
   export MEGATRON_PATH=/path/to/Megatron
   export DATASET_PATH=/path/to/dataset
   ```
