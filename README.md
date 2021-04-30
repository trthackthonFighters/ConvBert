# TensorTRT

## Requirements
* Python 3
* tensorflow 1.15
* numpy
* scikit-learn

### Export ONNX Model
```python convert2onnx_v2.py```

### Modify the Onehot Layer 
```python modify_onnx_gs.py ```

### Comparison between TF inference and TRT inference
```python test_tf_trt_infer.py```


# References

Here are some great resources we benefit:

Codebase: Our model codebase are based on [Convbert](https://github.com/yitu-opensource/ConvBert).

ConvBert: NeurIPS 2020 paper [ConvBERT: Improving BERT with Span-based Dynamic Convolution](https://arxiv.org/abs/2008.02496).

Dynamic convolution: [Implementation](https://github.com/pytorch/fairseq/blob/265791b727b664d4d7da3abd918a3f6fb70d7337/fairseq/modules/lightconv_layer/lightconv_layer.py#L75) from [Pay Less Attention with Lightweight and Dynamic Convolutions](https://openreview.net/pdf?id=SkVhlh09tX).
