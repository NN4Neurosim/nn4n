# nn4n.model.CTRNN.save()

[Back to CTRNN](https://github.com/zhaozewang/NN4Neurosci/blob/main/docs/model/CTRNN/index.md) </br>

## Introduction
Save the CTRNN model to a `.pth` file. All model parameters and attributes will be saved.

## Parameters
> `path` (str): required.
>> The path to save the model. Must end with `.pth`.

## Usage
```python
from nn4n.model import CTRNN

ctrnn = CTRNN()
ctrnn.save('ctrnn.pth')
```

see also: [load()](https://github.com/zhaozewang/NN4Neurosci/blob/main/docs/model/CTRNN/methods/load.md)
