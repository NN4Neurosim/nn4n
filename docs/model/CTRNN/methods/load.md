# nn4n.model.CTRNN.load()

[Back to CTRNN](https://github.com/zhaozewang/NN4Neurosci/blob/main/docs/model/CTRNN/index.md) </br>

## Introduction
Load the CTRNN model from a `.pth` file. All model parameters and attributes will be loaded.

## Parameters
> `path` (str): required.
>> The path to save the model. Must end with `.pth`.

## Usage
```python
from nn4n.model import CTRNN

ctrnn = CTRNN()
ctrnn.load('ctrnn.pth')
```

see also: [save()](https://github.com/zhaozewang/NN4Neurosci/blob/main/docs/model/CTRNN/methods/save.md)
