# FlashSampling

[![arXiv](https://img.shields.io/badge/arXiv-2603.15854-b31b1b.svg)](https://arxiv.org/abs/2603.15854) 
[![Website](https://img.shields.io/badge/Project-Website-blue)](https://flashsampling.github.io/FlashSampling) 
[![License](https://img.shields.io/badge/License-Apache_2.0-yellow.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/) 

### FlashSampling: Fast and Memory-Efficient Exact Sampling 

We present FlashSampling, an exact sampling primitive that fuses sampling into the LM-head matmul and never materializes the logits tensor in HBM. The method is simple: compute logits tile-by-tile on chip, add Gumbel noise, keep only one maximizer per row and per vocabulary tile, and finish with a small reduction over tiles. FlashSampling enables efficient categorical sampling by fusing the operation into the language model head matmul, eliminating memory overhead and reducing decoding time by up to 19%. 

**Author:** Tomas Ruiz\*, Zhen Qin\*, [Yifan Zhang†](https://yfz.ai), Xuyang Shen, Yiran Zhong, Mengdi Wang†

**Date:** February 28, 2026

[[Webpage](https://flashsampling.github.io/FlashSampling)] [[Huggingface](https://huggingface.co/papers/2603.15854)] 

![](./FlashSampling.png) 

## Citation

```bibtex
@article{ruiz2026flashsampling,
  title={FlashSampling: Fast and Memory-Efficient Exact Sampling},
  author = {Ruiz, Tomas and Qin, Zhen and Zhang, Yifan and Shen, Xuyang and Zhong, Yiran and Wang, Mengdi},
  journal={arXiv preprint arXiv:2603.15854},
  year={2026}
}
```

## 📜 License

This project is licensed under the [Apache License 2.0](https://www.google.com/search?q=LICENSE).
