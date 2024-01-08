# nn4n.mask.masks()

[Back to mask](https://github.com/zhaozewang/NN4Neurosci/blob/main/docs/mask/index.md) </br>

## Introduction
Return layer masks in a list.

## Returns
> `masks` (list):
>> A list of masks. Each mask is a 2D numpy array.

## Usage
```python
import torch
from nn4n.mask import MultiArea

params = {
    "hidden_size": 100,
    "input_dim": 2,
    "output_dim": 2,
}
multiarea = MultiArea(**params)
multiarea.masks()
```

Output
The first np.ndarray is the input mask, the second is the hidden mask, and the third is the readout mask.
```
[array([[1., 1.],
        [1., 1.],
        [1., 1.],
        [1., 1.],
        [1., 1.],
        [1., 1.],
          ...
        [1., 1.],
        [1., 1.],
        [1., 1.],
        [1., 1.],
        [1., 1.],
        [1., 1.],
        [1., 1.]]),
 array([[1., 1., 1., ..., 0., 0., 1.],
        [1., 1., 1., ..., 0., 0., 0.],
        [1., 1., 1., ..., 0., 0., 1.],
        ...,
        [0., 0., 0., ..., 1., 1., 1.],
        [0., 0., 0., ..., 1., 1., 1.],
        [0., 0., 1., ..., 1., 1., 1.]]),
 array([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
         1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
         1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
         1., 1.],
        [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
         1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
         1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
         1., 1.]])]
```
