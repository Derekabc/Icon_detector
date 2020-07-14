# Icon_detector
This repository is for the internship at T-Mobile, USA.

## Contents


# Export Trained Tensorflow Models
Edit the exporter.py file under object_detection file. We rename the original exporter.py as  exporter_frozen.py

```
# from workspace and run 
python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync.config --trained_checkpoint_prefix training/model.ckpt-197888 --output_directory output_inference_graph
```
Under the *output_inference_graph* folder we can folder the saved_model folder contains the trained model and variables.


