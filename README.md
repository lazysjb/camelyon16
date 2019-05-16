# camelyon16
### by Heinrich Peters & Seung-jae Bang

## Steps
Code is mainly structured to the following 5 steps.
User inputs such as I/O directories, zoom level configuration can be specified in `params.py`.
<br>

1. Create meta info of slides
- Example sh script:
```bash prepare_data.sh```
<br>

2. Create image partitions 
- Example sh script:
```bash partition_data.sh```
<br>

3. Model Code
- `models/Modelling.ipynb` contains the different types of architectures we ran the data on (ran on Google Colab)
<br>

4. Inference
- `models/single_input_inference.ipynb` and `models/double_input_inference.ipynb` for inference on test slides.
<br>

5. Evaluation
- `notebook/evaluate_output.ipynb` for example evaluation metrics on test slides
- `notebook/plot_heat_map.ipynb` for example heatmap plots on test / validation slides

## Replicating Inference Results
In order for our results to be replicable, we are hosting our trained model weights on Google Cloud.

#### VGG16 Single Input Transfer Learn Model
- Download: https://storage.googleapis.com/applied-dl-sj/camelyon/output_data/best_weights/vgg_zoom1_256_256_09-0.9554-0.1437.h5
- Example: Please refer to `models/single_input_inference.ipynb` for example of how to build & load the above weight to make inference.

#### VGG16 Double Input Transfer Learn Model
- Download: https://storage.googleapis.com/applied-dl-sj/camelyon/output_data/best_weights/combined_vgg_transfer_batch_norm_03-0.9627-0.1151.h5
- Example: Please refer to `models/double_input_inference.ipynb` for example of how to build & load the above weight to make inference.
