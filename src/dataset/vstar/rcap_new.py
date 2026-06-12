import sys
sys.path.append("/projects/torresani-lab/ajay/visual-grounded-vlm/")

import yaml
import argparse

import numpy as np
from transformers import AutoProcessor

from vllm import LLM, SamplingParams

def main(): 

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    model_path = cfg["model"]["path"]

    processor = AutoProcessor.from_pretrained(model_path)
    