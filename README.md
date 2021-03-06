# IncrementalVisualTracker-python
![](https://github.com/matkovst/IncrementalVisualTracker-python/blob/master/data/rocks-gif.gif)

This repository contains a python-implementation of Incremental Visual Tracking algorithm presented in the paper [Incremental Learning for Robust Visual Tracking](http://www.cs.toronto.edu/~dross/ivt/RossLimLinYang_ijcv.pdf).
The code was mostly grounded on the original MATLAB-implementation posted in http://www.cs.toronto.edu/~dross/ivt/.

## Requirements
1. Python 3.6+.
2. NumPy 1.16+.
3. Python-opencv 4.1+ (3.4+ may work as well).

## Run
`python demo.py --input <path_to_video> [--debug <int>] [--record <int>]`
