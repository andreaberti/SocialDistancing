# Social Distancing

Social distancing implies that people should physically distance themselves from one another, reducing close contact, and thereby reducing the spread of a contagious disease 

![Logo](https://github.com/ManuelPrandini/SocialDistancing/blob/master/img/logo.png)

## Final prototype

Sample street            |  Train Station
:-------------------------:|:-------------------------:
![](https://github.com/ManuelPrandini/SocialDistancing/blob/master/img/output_fasterRCNN_final_sample1.gif)  |  ![](https://github.com/ManuelPrandini/SocialDistancing/blob/master/img/output-fasterRCNN-final-sample52.gif)

## Structure of the project

- frames
  - sample1/ --> folder containing the frames of the video sample1
- video
  - sample1.mp4 --> input video1
  - sample1.txt --> information for video1
- models.py --> definition of object detection models and method for calculating social distance.
- trace_roi.py --> definition of the interface in open-cv to calculate the ROI 
- utils.py --> set of methods useful for the video process and for calculating the transformed points
- Predict_SocialDistancing.ipynb --> notebook to launch on the cloud (colab) to use the models on the gpu.

## Region of interest definition

Using first 4 points or coordinates for perspective transformation. The region marked by these 4 points are considered ROI. This polygon shaped ROI is then warped into a rectangle which becomes the bird eye view. The next 3 points are for horizontal and vertical unit length (in this case 180 cm)

![ROI](https://github.com/ManuelPrandini/SocialDistancing/blob/master/img/record_trace_ROI.gif)

## Pedestrian Detection Model

To achieve object detection of pedestrians on the scene we used various models. State-of-the-art object detectors use deep learning approaches, which are usually divided into two categories. The first one is called two-stage detectors, mostly based on R-CNN, which starts with region proposals and then performs the classification and bounding box regression. The second one is called one-stage detectors (for example YOLO, SSD, RetinaNet and EfficientDet). We have used the pre-trained of these models which perform very well. To use them we used 2 very popular computer vision frameworks on the market: Detectron2 for the Faster R-CNN model and ImageAI for the YOLOv3 and YOLOtiny models.

Detectron2            |  ImageAI
:-------------------------:|:-------------------------:
![](https://miro.medium.com/max/4000/0*VbMjGBHMC6GnDKUp.png)  |  ![](https://gitee.com/vincent_hice/ImageAI/raw/master/logo1.png)

At runtime the user can select the model with which to perform the object detection. YOLOv3 allows fast execution with good accuracy while Faster R-CNN provides better accuracy but higher computation times. The models are defined as follows:

```python
def YoloV3_model(yolov3_model_path):
    '''
    Method that creates a YoloV3 model, using config from ImageAi core library.
    :param yolov3_model_path: the path of the config file
    :return: the YoloV3 model used to predict and the custom objects ( only poeple) to pass to the model
    during prediction.
    '''
    detector = ObjectDetection()
    detector.setModelTypeAsYOLOv3()  # Se vuoi usare yolo tiny cambia il set model
    detector.setModelPath(yolov3_model_path)
    custom_objects = detector.CustomObjects(person=True)
    detector.loadModel()
    return detector, custom_objects
    
def faster_RCNN_model():
    '''
    Method that creates a fasterRCNN model, using config and pretrained weights
    from detectron2 core library
    :return: the fasterRCNN model used to predict
    '''
    cfg = get_cfg()
    cfg.MODEL.DEVICE = 'cuda'

    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_C4_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.94  # set threshold for this model

    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_C4_3x.yaml")
    predictor = DefaultPredictor(cfg)
    return predictor
```

## Authors

- [Prandini Manuel](https://github.com/ManuelPrandini)

- [Bianca Francesco ](https://github.com/francescobianca)

- [Berti Andrea ](https://github.com/andreaberti)

## References

[1]. Dongfang Yang, Ekim Yurtsever, Vishnu Renganathan, Keith A. Redmill, Ümit Özgüner, “A Vision-based Social Distancing and Critical Density Detection System for COVID-19”, 2020.

[2]. Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun, “Faster R-CNN: Towards Real-Time Object
Detection with Region Proposal Networks”, 2016.

[3]. Joseph Redmon, Ali Farhadi, “YOLOv3: An Incremental Improvement”.

[4]. Pranav Adarsh, Pratibha Rathi, Manoj Kumar, “YOLO v3-Tiny: Object Detection and Recognition using
one stage improved model”, 2020 6th International Conference on Advanced Computing & Communication Systems (ICACCS). 

[5]. Yuxin Wu and Alexander Kirillov and Francisco Massa and Wan-Yen Lo and Ross Girshick, "Detectron2", 2019, https://github.com/facebookresearch/detectron2

[6]. Moses and John Olafenwa, "ImageAI, an open source python library built to empower developers to build applications and systems  with self-contained Computer Vision capabilities", mar 2018, https://github.com/OlafenwaMoses/ImageAI
