# camelyon16
### by Heinrich Peters & Seung-jae Bang

## Steps
Code is mainly structured to the following 5 steps:
<br>

1. Create meta info of slides
- Example sh script:
```bash prepare_data.sh```
<br>

2. Create image partitions 
- Example sh script:
```bash partition_data.sh```

User inputs such as I/O directories, zoom level configuration can be specified in `params.py`
<br>

3. Model Code
- `models/Modelling.ipynb` contains the different types of architectures we ran the data on (ran on Google Colab)
<br>

4. Inference


## Replicating Inference Results
In order for our results to be replicable, we are hosting our trained model weights on Google Cloud.

#### VGG16 Single Input Transfer Learn Model

