import os
from ultralytics import YOLO
import json
import matplotlib.pyplot as plt 
from pathlib import Path

# Replace with the path to your top-level folder
root_dir = Path("big_images/")

# Find all .jpg files recursively under the root folder
jpg_paths = list(root_dir.rglob("*.jpg"))

# Convert to strings (optional, for printing or further processing)
jpg_paths = [str(path) for path in jpg_paths]

# Print or use the list
# for path in jpg_paths:
#     print(path)
print(len(jpg_paths))


exp_names = {
    'color': ['beige', 'bleu'],
    'typo': ['boucharde', 'flamme', 'spuntato'],
    'state': ['degrade', 'satisfaisant'],
    # 'cs': ['beige_degrade', 'beige_satisfaisant', 'bleu_degrade', 'bleu_satisfaisant'],
    # 'ct': ['beige_boucharde', 'beige_flamme', 'beige_spuntato', 'bleu_boucharde', 'bleu_flamme', 'bleu_spuntato'],
    # 'ts': ['boucharde_degrade', 'boucharde_satisfaisant', 'flamme_degrade', 'flamme_satisfaisant', 'spuntato_degrade', 'spuntato_satisfaisant']            
}

import gc
import torch
from ultralytics import YOLO
from yolo_to_vgg import yolo_results_to_vgg

# Option A: hide GPU0 at the shell before running (recommended)
# export CUDA_VISIBLE_DEVICES=1,2,3
# Then use device="0" inside Python (they will be physical 1,2,3)
#
# Option B: explicit device string (works with ultralytics device arg)
# device = "1,2,3"   # avoid GPU 0

bs = 64

for exp in exp_names.keys():
    model = YOLO(f"nancy_v2/pave_stanislas_{exp}/weights/best_{exp}.pt", task="segment")
    

    # optional: move model to CUDA (Ultralytics usually handles it, but explicit is OK)
    # model.to("cuda:0")  # only if CUDA_VISIBLE_DEVICES was used to remap GPUs
    # NOTE: when using device="1,2,3" below, ultralytics handles device placement.

    # inference inside no_grad to reduce graph / memory
    
    results = model(
        ba_paths,
        imgsz=512,
        save=True,
        retina_masks=True,
        agnostic_nms=True,
        show_conf=False,
        batch = 32,
        conf=.25,
        iou=.5,
        verbose=True,
        device=0,
        stream=True
    )

   
    yolo_results_to_vgg(results, model.names, f"big_test_{exp}.json", offset=128)

    # final cleanup per-experiment (if many experiments)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass