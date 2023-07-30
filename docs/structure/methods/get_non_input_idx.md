# nn4n.structure.get_non_input_idx()

[Back to structure](https://github.com/zhaozewang/NN4Neurosci/blob/main/docs/structure/index.md) </br>

## Introduction
Get the indices of the neurons that receive inputs.

## Returns
> `non_input_idx` (np.ndarray):
>> The indices of neurons that don't receive inputs.

## Usage
The example below will create a `MultiArea` object with 3 areas, where the first area is the input area. The `get_non_input_idx()` method will return indices of neurons in area 2 and 3, as they don't receive inputs.

```python
import torch
from nn4n.structure import MultiArea

params = {
    "n_areas": 3,
    "area_connectivities": [0.1, 0.1],
    "hidden_size": 90,
    "input_dim": 1,
    "output_dim": 1,
    "input_areas": [0],
    "readout_areas": [2],
}
multiarea = MultiArea(**params)
multiarea.get_non_input_idx()
```

Output
```
array([30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
       47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
       64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
       81, 82, 83, 84, 85, 86, 87, 88, 89])
```

see also: [get_readout_idx()](https://github.com/zhaozewang/NN4Neurosci/blob/main/docs/structure/methods/get_readout_idx.md) | [get_input_idx()](https://github.com/zhaozewang/NN4Neurosci/blob/main/docs/structure/methods/get_input_idx.md)
