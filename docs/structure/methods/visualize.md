# nn4n.structure.visualize()

[Back to structure](https://github.com/zhaozewang/NN4Neurosci/docs/structure/index.md) </br>

## Introduction
Print out the generated mask.

## Usage
The example below will create a `MultiArea` object with 3 areas. `structure.visualize()` is called to print out the generated masks.

```python
area_connectivities = np.array([
    [1.0, 0.1, 0.02],
    [0.1, 1.0, 0.1],
    [0.02, 0.1, 1.0]
])
n_areas = [10, 50, 40]
params = {
    "n_areas": n_areas,
    "area_connectivities": area_connectivities,
    "hidden_size": 100,
    "input_dim": 2,
    "output_dim": 2,
    "input_areas": [0],
    "readout_areas": [1],
}

multiarea = MultiArea(**params)
multiarea.visualize()
```

Output

<img src="https://github.com/zhaozewang/NN4Neurosci/docs/images/structure/input_mask.png" width="500"></br>
<img src="https://github.com/zhaozewang/NN4Neurosci/docs/images/structure/hidden_mask.png" width="500"></br>
<img src="https://github.com/zhaozewang/NN4Neurosci/docs/images/structure/readout_mask.png" width="500"></br>

see also: [get_readout_idx()](https://github.com/zhaozewang/NN4Neurosci/docs/structure/methods/get_readout_idx.md) | [get_input_idx()](https://github.com/zhaozewang/NN4Neurosci/docs/structure/methods/get_input_idx.md) | [get_non_input_idx()](https://github.com/zhaozewang/NN4Neurosci/docs/structure/methods/get_non_input_idx.md)
