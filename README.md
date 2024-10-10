# Face-Police

Object Detection app specifically for Face Masks. 
Currently it can detect for the following classes:
1. No masks
2. Incorrect wearing of mask 
3. Correct mask fit

## Installation and Running
Together with the app should be the model file in **output/**.

The app can be run almost out of the box by first installing the python dependencies with pip.

```
pip install -r requirements.txt
```

Next simply run the app using 
```
python main.py
```

## Retraining and customization

Alot of things can be controlled by checking **config.py**, things such as directories can be edited, or training proportions.


### Dataset Acquisition

This project was built using this dataset:
https://www.kaggle.com/api/v1/datasets/download/andrewmvd/face-mask-detection

You can download the contents and place them into the directory indicated by `DATASET_DIR` inside the **config.py** file.

### Preprocessing
Preprocessing can be achieved by running the **preprocess.py** script. It will make additional directories and files for YOLO to use.

> As of this version, manual intervention is needed for YOLO, the **data.yaml** must be manually edited to point to the correct location. Additonally, you may need to configure the ultralytics config file.


### Training
Training can be done with the **train.py** script. Simply run it after the preprocessing stage.

Some paramters regarding training can also be controlled in the config file.

Once training has completed, the model will be in a directory similar to *"runs/train"*.

Simply bring it over into the output folder.
