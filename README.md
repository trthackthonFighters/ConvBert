# TensorTRT

## Requirements
* Python 3
* tensorflow 1.15
* numpy
* scikit-learn

### Export ONNX Model
```python convert2onnx_v2.py```

### Modify the Onehot Layer 
```python modify_onnx_gs.py```

### Build the Onehot plugin from Nvidia Hackathon Repo
```
git clone https://github.com/NVIDIA/trt-samples-for-hackathon-cn.git
cd build
make
```
and copy Onehot_plugin.so to Convbert folder.

### Comparison between TF inference and TRT inference
```python test_tf_trt_infer.py```

### Result
```
tf_time= [INFO] TF  execution time 367.3338 ms
trt_time= TRT execution time 9.16735 ms
```
The value is the average of inference time. The tf_time is over-estimated as it may contain the cpu time. It may need to do profiling to get an accurate value.

The speed-up ratio is 40.06.


# References

Here are some great resources we benefit:

Codebase: Our model codebase are based on [Convbert](https://github.com/yitu-opensource/ConvBert).

ConvBert: NeurIPS 2020 paper [ConvBERT: Improving BERT with Span-based Dynamic Convolution](https://arxiv.org/abs/2008.02496).

Dynamic convolution: [Implementation](https://github.com/pytorch/fairseq/blob/265791b727b664d4d7da3abd918a3f6fb70d7337/fairseq/modules/lightconv_layer/lightconv_layer.py#L75) from [Pay Less Attention with Lightweight and Dynamic Convolutions](https://openreview.net/pdf?id=SkVhlh09tX).
