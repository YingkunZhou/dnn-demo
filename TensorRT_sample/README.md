
# [DLA and GPU cores at the same time](https://forums.developer.nvidia.com/t/dla-and-gpu-cores-at-the-same-time/122287)

Here is a sample to run GPU and DLAs at the same time.

1. Please prepare TensorRT engine of GPU and DLA with trtexec first.

For example,

```bash
$ /usr/src/tensorrt/bin/trtexec --onnx=/usr/src/tensorrt/data/mnist/mnist.onnx --saveEngine=gpu.engine
$ /usr/src/tensorrt/bin/trtexec --onnx=/usr/src/tensorrt/data/mnist/mnist.onnx --useDLACore=0 --allowGPUFallback --saveEngine=dla.engine
```

2. Compile

[TensorRT_sample.zip](https://forums.developer.nvidia.com/uploads/short-url/hyv6UFFVbIWi9baWbfBzr4Ud3zK.zip) (3.5 KB)

```bash
$ unzip TensorRT_sample.zip
$ cd TensorRT_sample
$ make
```

3. Test

Please put the gpu.engine and dla.engine generated in step1 to the *TensorRT_sample*.

The command runs like this

```bash
$ ./test  <thread0> <thread1> <thread2>, ..., <threadK>  # -1: GPU, 0: DLA0, 1: DLA1
```

Ex. Run GPU+DLA0+DLA1.

```bash
$ ./test -1 0 1
```

The 11 TFLOPS use all the GPU cores: including 512 CUDA cores and 64 Tensor Cores.

Each DLA can reach 2.5TFLOPS.

It’s possible that use all the processor at the same time.
You can check our GitHub as an example to do so:

[Jetson Benchmark](https://github.com/NVIDIA-AI-IOT/jetson_benchmarks)