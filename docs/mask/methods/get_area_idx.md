# nn4n.mask.get_area_idx()

[Back to mask](https://github.com/zhaozewang/NN4Neurosci/blob/main/docs/mask/index.md) </br>

## Introduction
Get the node indices of a specific area.

## Parameters
> `area` (str) or (int): required.
>> The name of the area or the index of the area.

## Returns
> `indices` (np.ndarray):
>> The indices of the nodes in the specified area.

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
multiarea.get_areas()  # ['area_1', 'area_2']
multiarea.get_area_idx('area_1')
```

Output
```
array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
       34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49])
```

see also: [get_areas()]('https://github.com/zhaozewang/NN4Neurosci/blob/main/docs/mask/methods/get_areas.md')
