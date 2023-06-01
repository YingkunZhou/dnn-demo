
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

## chatgpt read code

This code appears to be a C++ implementation of a multi-task inference system. It also includes CUDA and TensorRT components, which are popular tools for deep learning inference.

The code defines a class called `Task` which encapsulates a TensorRT plan and provides a method called `exec_once()` which runs a single inference forward pass. Each Task can be executed on a GPU or DLA core by specifying it as an argument on the class constructor.

The Task class also has a method called `repeatWithSync()` which takes a `SyncType` parameter and a lambda function to initialize the worker thread. `SyncType` specifies how the worker threads should synchronize after each inference forward pass, either through a CUDA stream or a CUDA event. This method creates a worker thread that repeatedly calls `exec_once()`, waits for synchronization, and updates a counter to determine inference rate.

The `main()` function creates multiple instances of the Task class using plan files, one for each GPU or DLA core specified as command line arguments. It then creates worker threads for each Task using the `repeatWithSync()` method. Finally, `main()` enters a loop that constantly prints the inference rate of each Task until the program is stopped with a `SIGINT` signal. When this happens, the program joins all the worker threads and ends.

> SIGINT signal要怎么输入呢

在 Unix 系统中，可以使用 `Ctrl+C` 组合键来发送 `SIGINT` 信号给正在运行的程序。在程序运行期间，按下 `Ctrl+C`，程序会收到 `SIGINT` 信号并停止运行。

语句 `while (!Task::shouldStop()) {`
是什么意思

这个语句是一个循环语句，循环内的代码会一直执行，直到 `Task::shouldStop()` 的返回值为 true。

在这个程序中，`Task::shouldStop()`是一个静态方法，它返回了类变量 `gStop` 的值，该值会在接收到 `SIGINT` 信号时被设置为 true。因此，当这个方法返回 false 时，循环内的代码就会一直执行，每秒钟打印当前每个 Task 的 inference rate，直到收到 `SIGINT` 信号，然后程序跳出这个循环，停止所有的 worker threads。