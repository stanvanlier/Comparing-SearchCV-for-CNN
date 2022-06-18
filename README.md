# Comparing-SearchCV-for-CNN

Choose which experiments to run and import (option 1). Alternatively, the list
can be defined in the notebook itself (option 2). Or run it locally (option 3).

## Option 1: Using one of the predefined [experiments](experiments/)

Paste the following in a notebook and replace `X` by the number of
experiments to run.

### For CPU:

```python
!pip install numpyencoder
!pip install sklearn-genetic-opt

!git clone https://github.com/stanvanlier/Comparing-SearchCV-for-CNN.git
import sys
sys.path.append('Comparing-SearchCV-for-CNN')

from src.data.datasets import download
download()

from src.run.cpu import main 
from experiments.expsX import experiments

main(experiments, batch_size=256)
```

You can see a sample notebook in [notebooks/cpu.ipynb](notebooks/cpu.ipynb).

### For GPU:

```python
!pip install numpyencoder
!pip install sklearn-genetic-opt

!git clone https://github.com/stanvanlier/Comparing-SearchCV-for-CNN.git
import sys
sys.path.append('Comparing-SearchCV-for-CNN')

from src.data.datasets import download
download()

from src.run.gpu import main 
from experiments.expsX import experiments

main(experiments, batch_size=512)
```

You can see a sample notebook in [notebooks/gpu.ipynb](notebooks/gpu.ipynb).

### For TPU:

```python
!pip install cloud-tpu-client==0.10 torch==1.11.0 https://storage.googleapis.com/tpu-pytorch/wheels/colab/torch_xla-1.11-cp37-cp37m-linux_x86_64.whl

!pip install numpyencoder
!pip install sklearn-genetic-opt

!git clone https://github.com/stanvanlier/Comparing-SearchCV-for-CNN.git
import sys
sys.path.append('Comparing-SearchCV-for-CNN')

from src.data.datasets import download
download()

from src.run.tpu import main 
from experiments.expsX import experiments

main(experiments, batch_size=1024)
```

You can see a sample notebook in [notebooks/tpu.ipynb](notebooks/tpu.ipynb).

### Pulling new versions without restarting runtime

```python
!cd Comparing-SearchCV-for-CNN/ && git pull && cd ..

import importlib
importlib.reload(experiments.expsX)
#importlib.reload(src.run.tpu)
```

## Option 2: Defining a new experiments list

Instead of using `from experiments.expsX import experiments`, the `experiments`
variable can be defined in the notebook. For example, copy the contents of
[experiments/exps1.py](experiments/exps1.py) into a new cell.


## Option 3: Run locally

### Setup

Clone this repo:

```
git clone https://github.com/stanvanlier/Comparing-SearchCV-for-CNN.git
cd Comparing-SearchCV-for-CNN
```

Setup a new virtual environment with:

```
python3 -m venv venv
source venv/bin/active
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
pip install numpy pandas sklearn
pip install numpyencoder
pip install sklearn-genetic-opt
python download.py
```

### Run

```
source venv/bin/active
python main.py
```
