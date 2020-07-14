import tensorflow as tf

# Assuming object detection API is available for use
from object_detection.utils.config_util import create_pipeline_proto_from_configs
from object_detection.utils.config_util import get_configs_from_pipeline_file
from object_detection import exporter
# Configuration for model to be exported
config_pathname = 'training/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync.config'

# Input checkpoint for the model to be exported
# Path to the directory which consists of the saved model on disk (see above)
trained_model_dir = 'training/'

# Create proto from model confguration
configs = get_configs_from_pipeline_file(config_pathname)
pipeline_proto = create_pipeline_proto_from_configs(configs=configs)

# Read .ckpt and .meta files from model directory
checkpoint = tf.train.get_checkpoint_state(trained_model_dir)
input_checkpoint = checkpoint.model_checkpoint_path

# Model Version
model_version_id = '1'

# Output Directory
output_directory = './' + str(model_version_id)

# Export model for serving
object_detection.exporter.export_inference_graph(input_type='image_tensor',pipeline_config=pipeline_proto,trained_checkpoint_prefix=input_checkpoint,output_directory=output_directory)