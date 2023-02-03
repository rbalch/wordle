import os, shutil
from utils.wordle_dict import all_words, words

master_dict = all_words + words
local_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_file = os.path.join(local_dir, 'wordle_neat_config.ini')
out_dir = os.path.join(local_dir, 'out')


def leaky_relu6(x):
    return min(x, 6) if x > 0 else 0.01 * x


def clean_output():
    if os.path.isdir(out_dir):
        # remove files from previous run
        shutil.rmtree(out_dir)
    # create the output directory
    os.makedirs(out_dir, exist_ok=False)
    with open(os.path.join(out_dir, '__init__.py'), 'w') as f:
        pass
