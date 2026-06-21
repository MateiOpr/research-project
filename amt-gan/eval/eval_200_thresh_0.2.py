import sys
sys.argv = ['eval', '--save_path', 'assets/datasets/save_200_thresh_0.2', '--target_path', 'assets/datasets/test/047073.jpg']
from test import attack_local_models
attack_local_models(attack=True)
