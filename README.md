# Icon_detector
This repository is for the internship at T-Mobile, USA.

## Contents


# Build an icon Detector based on Tensorflow Object Detection API


# Export Trained Tensorflow Models
## Environment Setup.
- python=2.7
- tensorflow==1.14.0
- tensorflow-estimator==1.14.0 

```
conda create -n py2.7 pip python=2.7
source activate py2.7
pip install tensorflow==1.14.0
pip install matplotlib
pip install pillow
pip install tensorflow-estimator
cd object_detection
protoc protos/*.proto --python_out=.
```

Edit the exporter.py file under object_detection folder. We renamed the original exporter.py as  exporter_frozen.py

```
# from workspace and run 
python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync.config --trained_checkpoint_prefix training/model.ckpt-197888 --output_directory output_inference_graph
```
Under the *output_inference_graph* folder we can folder the saved_model folder contains the trained model and variables.


