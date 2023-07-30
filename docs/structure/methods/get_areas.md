# nn4n.structure.get_areas()

[Back to structure](https://github.com/zhaozewang/NN4Neurosci/docs/structure/index.md) </br>

## Introduction
Retrieve the area names

## Returns
> `area_names` (list):
>> A list of area names.

## Usage
```python
import torch
from nn4n.structure import MultiArea

params = {
    "hidden_size": 100,
    "input_dim": 2,
    "output_dim": 2,
}
multiarea = MultiArea(**params)
multiarea.get_areas()
```

Output
```
['area_1', 'area_2']
```

see also: [get_area_idx()]('https://github.com/zhaozewang/NN4Neurosci/docs/structure/methods/get_area_idx.md')
