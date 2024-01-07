import os

from voice_detect.utils.gen_utils import create_dir

# custom model directory name
save_model_dir = 'saved_models'
cnn_model_dir = 'cnn'
ffnn_model_dir = 'ffnn'

# get root directory and other directories
root_dir = os.path.dirname(os.path.abspath(__file__))
abs_cnn_model_path = os.path.join(root_dir, save_model_dir, cnn_model_dir)
abs_ffnn_model_path = os.path.join(root_dir, save_model_dir, ffnn_model_dir)


create_dir(abs_cnn_model_path)
create_dir(abs_ffnn_model_path)
