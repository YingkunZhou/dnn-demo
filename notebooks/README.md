```bash
pip install --upgrade jupyter ipywidgets
```

# [Accelerate Deep Learning Models using Torch-TensorRT](https://github.com/pytorch/TensorRT/blob/main/notebooks/qat-ptq-workflow.ipynb)

<a href="https://colab.research.google.com/github/YingkunZhou/dnn-demo/blob/main/notebooks/qat-ptq-workflow.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

观察：
1. 3轮的transfer learning预训练对于精度的影响极大，在74%到84%之间；训练一轮的时间在orin上是1分半左右
2. 目前这一版本的PTQ和QAT精度并没有差的那么大，所以考虑到QAT的2轮微调训练需要在orin上耗时5分多钟，而且推理的性能也没有PTQ高（可是为什么呢？），所以PTQ反而是更香的。

# [efficientformer demo](https://github.com/pytorch/TensorRT/blob/main/notebooks/efficientformer.ipynb)

<a href="https://colab.research.google.com/github/YingkunZhou/dnn-demo/blob/main/notebooks/efficientformer.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

疑点：