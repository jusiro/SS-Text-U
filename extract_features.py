"""
Main function for feature extraction.
"""

import argparse
import torch
import os
import time
import datetime
import numpy as np

from torch.multiprocessing import set_start_method

from transformers import AutoModel, AutoTokenizer, logging
from huggingface_hub import PyTorchModelHubMixin

logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from documents.local_data.constants import PATH_FEATURES, PATH_DATASETS, PATH_DATAFRAMES, PATH_VLMS_WEIGHTS
from data.configs.tasks_batch import * 
from data.configs.tasks_configs import get_task_cfg, get_split_cfg
from vlms.prompts.text_embedder import get_text_prototypes
from data.datagen.dataloader import set_loader
from vlms.utils import extract_vision_features, predict_from_features

from libs.archs.lp import LinearSoftmax

# Device for training/inference
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Set seeds for reproducibility
from utils.misc import set_seeds
set_seeds(42, use_cuda=device == 'cuda')


def process(args):

    # %---------------------------------------------------------

    # Create folder to save features.
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    # Loop over all datasets.
    for task in args.tasks:
        print("Processing: {:10s}".format(task["name"]))

        # Retrieve task splits (train, test).
        splits_cfg = get_task_cfg(task["name"])

        # Get specific experiment settings (i.e. dataframe path, classes, tasks, ...).
        task_cfgs = get_split_cfg(splits_cfg["train"])

        # Get pre-trained VLM.
        if task["vlm"] == "conch":
            from conch.open_clip_custom import create_model_from_pretrained
            model, _ = create_model_from_pretrained('conch_ViT-B-16', checkpoint_path=PATH_VLMS_WEIGHTS + "conch.bin")
            model.to(device).float()
            logit_scale = model.logit_scale.item()
        elif task["vlm"] == "flair":
            # Load pre-trained model
            from flair import FLAIRModel
            model = FLAIRModel.from_pretrained("jusiro2/FLAIR")
            model.to(device).float().eval()
            # Avoit unit hyper-sphere feature normalization (will be done latter)
            model.vision_model.projection_head_vision.norm_projection = False
            model.text_model.projection_head_text.norm_projection = False
            model.vision_model.projection_head_vision.norm_modality = False
            model.text_model.projection_head_text.norm_modality = False
            # Retrieve logit scale
            logit_scale = model.logit_scale.item()
        elif task["vlm"] == "convirt":
            # Load pre-trained model
            from dlilp import VLMModel
            model = VLMModel.from_pretrained("jusiro2/CONVIRT")
            model.to(device).float().eval()
            # Avoit unit hyper-sphere feature normalization (will be done latter)
            model.vision_model.projection_head_vision.norm_projection = False
            model.text_model.projection_head_text.norm_projection = False
            model.vision_model.projection_head_vision.norm_modality = False
            model.text_model.projection_head_text.norm_modality = False
            # Retrieve logit scale
            logit_scale = model.logit_scale.item()
        else:
            print("VLM not available....")
            return

        # Generate zero-shot prototypes
        textual_prototypes = get_text_prototypes(model, task_cfgs["targets"], vlm=task["vlm"]).cpu().numpy()

        # Set zero-shot weights for linear classifier (for producing zero-shot logits)
        head = LinearSoftmax(D=textual_prototypes.shape[-1], C=textual_prototypes.shape[0], norm=True, 
        weights=textual_prototypes, logit_scale=logit_scale)

        # Set training and test datasets
        datasets = [splits_cfg["train"]] + [splits_cfg["test"]]

        experiment = {"partitions": {}}
        # Set splits datasets
        for i in range(len(datasets)):
            experiment["partitions"][i] = {}
            experiment["partitions"][i]["domain"] = {datasets[i]}
            # Get specific experiment settings (i.e. dataframe path, classes, tasks, ...)
            split_cfgs = get_split_cfg(datasets[i])
            experiment["partitions"][i]["dataloader"] = set_loader(
                PATH_DATAFRAMES + split_cfgs["dataframe"], args.data_root_path + split_cfgs["base_samples_path"],
                split_cfgs["targets"], size=task["img_size"], norm=task["img_norm"])

        # Extract feature for different splits
        time_extraction = []
        for i_domain in range(0, len(experiment["partitions"])):
            print("  Processing: [{dataset}]".format(dataset=datasets[i_domain]))

            # Extract vision features
            time_adapt_i_1 = time.time()
            id = args.out_path + datasets[i_domain] + "_" + task["vlm"].lower().replace("/", "_")
            if not os.path.isfile(id + ".npz"):
                print("  Extracting features and saving in disk")
                if task["vlm"] == "conch":
                    feats_ds, refs_ds = extract_vision_features(
                        model.visual, experiment["partitions"][i_domain]["dataloader"])
                elif task["vlm"] == "flair":
                    feats_ds, refs_ds = extract_vision_features(
                        model.vision_model, experiment["partitions"][i_domain]["dataloader"])
                elif task["vlm"] == "convirt":
                    feats_ds, refs_ds = extract_vision_features(
                        model.vision_model, experiment["partitions"][i_domain]["dataloader"])

                print("  Extracting logits")
                logits_ds = predict_from_features(head, torch.tensor(feats_ds), bs=args.bs, act=False, epsilon=1.0)
                logits_ds = logits_ds.cpu().numpy()
                time_adapt_i_2 = time.time()

                print("  Saving in disk")
                np.savez(id, feats_ds=feats_ds, logits_ds=logits_ds, refs_ds=refs_ds,
                         logit_scale=model.logit_scale.item(), textual_prototypes=textual_prototypes)
            else:
                time_adapt_i_2 = time.time()
            time_adapt_i = time_adapt_i_2 - time_adapt_i_1
            time_extraction.append(time_adapt_i)
            print(str("Feature extraction time: " + str(datetime.timedelta(seconds=time_adapt_i))))
    print("Average time: " + str(datetime.timedelta(seconds=np.mean(time_extraction))))


def main():

    # Avoid multiple workers multiprocessing errors
    set_start_method('spawn')

    parser = argparse.ArgumentParser()

    # Tasks
    parser.add_argument('--tasks', default=TASKS)

    # Folders, data, etc.
    parser.add_argument('--data_root_path', default=PATH_DATASETS)
    parser.add_argument('--out_path', default=PATH_FEATURES, help='output path')

    # Other hyper-param
    parser.add_argument('--bs', default=128, type=int, help='Batch size')

    args, unknown = parser.parse_known_args()

    process(args=args)


if __name__ == "__main__":
    main()