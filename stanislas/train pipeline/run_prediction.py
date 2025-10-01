# import os
# from ultralytics import YOLO
# import json
# import matplotlib.pyplot as plt 
# from pathlib import Path

# # Replace with the path to your top-level folder
# root_dir = Path("big_images/")

# # Find all .jpg files recursively under the root folder
# jpg_paths = list(root_dir.rglob("*.jpg"))

# # Convert to strings (optional, for printing or further processing)
# jpg_paths = [str(path) for path in jpg_paths]

# # Print or use the list
# # for path in jpg_paths:
# #     print(path)
# len(jpg_paths)


# exp_names = {
#     'color': ['beige', 'bleu'],
#     'typo': ['boucharde', 'flamme', 'spuntato'],
#     'state': ['degrade', 'satisfaisant'],
#     # 'cs': ['beige_degrade', 'beige_satisfaisant', 'bleu_degrade', 'bleu_satisfaisant'],
#     # 'ct': ['beige_boucharde', 'beige_flamme', 'beige_spuntato', 'bleu_boucharde', 'bleu_flamme', 'bleu_spuntato'],
#     # 'ts': ['boucharde_degrade', 'boucharde_satisfaisant', 'flamme_degrade', 'flamme_satisfaisant', 'spuntato_degrade', 'spuntato_satisfaisant']            
# }

# import gc
# import torch
# from ultralytics import YOLO
# from yolo_to_vgg import yolo_results_to_vgg

# # Option A: hide GPU0 at the shell before running (recommended)
# # export CUDA_VISIBLE_DEVICES=1,2,3
# # Then use device="0" inside Python (they will be physical 1,2,3)
# #
# # Option B: explicit device string (works with ultralytics device arg)
# # device = "1,2,3"   # avoid GPU 0

# bs = 8

# for exp in exp_names.keys():
#     model = YOLO(f"nancy_v2/pave_stanislas_{exp}/weights/best_{exp}.pt", task="segment")
#     results = []

#     # optional: move model to CUDA (Ultralytics usually handles it, but explicit is OK)
#     # model.to("cuda:0")  # only if CUDA_VISIBLE_DEVICES was used to remap GPUs
#     # NOTE: when using device="1,2,3" below, ultralytics handles device placement.

#     for nb in range(0, len(jpg_paths), bs):
#         batch_paths = jpg_paths[nb:nb+bs]
#         print(f"[{exp}] processing {nb} -> {nb+len(batch_paths)-1}")

#         # inference inside no_grad to reduce graph / memory
#         with torch.no_grad():
#             batch_results = model(
#                 batch_paths,
#                 imgsz=512,
#                 save=True,
#                 retina_masks=True,
#                 agnostic_nms=True,
#                 show_conf=False,
#                 conf=.25,
#                 iou=.5,
#                 verbose=False,
#                 device=0,
#             )

#         batch_results = [r.cpu() for r in batch_results]
        
#         # collect results (store only required info to save memory)
#         results.extend(batch_results)

#         # ---- explicit cleanup to free CUDA memory ----
#         # delete references you won't need (results list keeps only Results objects)
#         del batch_results
#         # run python GC
#         gc.collect()
#         # ensure all CUDA kernels finished (optional, sometimes helpful)
#         if torch.cuda.is_available():
#             try:
#                 # synchronize and release cached memory
#                 torch.cuda.synchronize()
#             except Exception:
#                 pass
#             # release PyTorch cached allocations to the driver
#             torch.cuda.empty_cache()
#             # attempt to collect orphaned CUDA IPC memory segments
#             # available in recent PyTorch versions
#             try:
#                 torch.cuda.ipc_collect()
#             except Exception:
#                 pass

#         # optional: print a short memory summary (helpful for debugging)
#         if torch.cuda.is_available():
#             # show memory allocated on the first visible device
#             pass
#             # print(torch.cuda.memory_summary())

#     # write final json
#     yolo_results_to_vgg(results, model.names, f"big_test_{exp}.json", offset=128)

#     # final cleanup per-experiment (if many experiments)
#     del results, model
#     gc.collect()
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#         try:
#             torch.cuda.ipc_collect()
#         except Exception:
#             pass


# import os
# from ultralytics import YOLO
# import gc
# import torch
# from pathlib import Path
# from yolo_to_vgg import yolo_results_to_vgg

# # Dossier racine
# root_dir = Path("big_images/")

# # Extensions d’images acceptées
# jpg_paths = [str(p) for p in root_dir.rglob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
# print("Total images found:", len(jpg_paths))

# exp_names = {
#     'color': ['beige', 'bleu'],
#     'typo': ['boucharde', 'flamme', 'spuntato'],
#     'state': ['degrade', 'satisfaisant'],
# }

# bs = 64          # taille de batch
# chunk_size = 10000  # taille des sous-blocs

# for exp in exp_names.keys():
#     print(f"\n=== Running experiment: {exp} ===")
#     model = YOLO(f"nancy_v2/pave_stanislas_{exp}/weights/best_{exp}.pt", task="segment")
#     results = []

#     # on découpe en sous-blocs de 10k images
#     for chunk_start in range(0, len(jpg_paths), chunk_size):
#         chunk_paths = jpg_paths[chunk_start:chunk_start + chunk_size]
#         print(f"[{exp}] Processing chunk {chunk_start} -> {chunk_start+len(chunk_paths)-1}")

#         for nb in range(0, len(chunk_paths), bs):
#             batch_paths = chunk_paths[nb:nb+bs]
#             print(f"[{exp}]   batch {nb} -> {nb+len(batch_paths)-1}")

#             with torch.no_grad():
#                 batch_results = model(
#                     batch_paths,
#                     imgsz=512,
#                     save=True,
#                     retina_masks=True,
#                     agnostic_nms=True,
#                     show_conf=False,
#                     batch=16,
#                     conf=.25,
#                     iou=.5,
#                     verbose=True,
#                     device=0,
#                 )

#             batch_results = [r.cpu() for r in batch_results]
#             results.extend(batch_results)

#             # cleanup mémoire
#             del batch_results
#             gc.collect()
#             if torch.cuda.is_available():
#                 try:
#                     torch.cuda.synchronize()
#                 except Exception:
#                     pass
#                 torch.cuda.empty_cache()
#                 try:
#                     torch.cuda.ipc_collect()
#                 except Exception:
#                     pass

#     # Sauvegarde des résultats en JSON
#     yolo_results_to_vgg(results, model.names, f"big_test_{exp}.json", offset=128)

#     # cleanup modèle
#     del results, model
#     gc.collect()
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#         try:
#             torch.cuda.ipc_collect()
#         except Exception:
#             pass

import os
import gc
import sys
import torch
from pathlib import Path
from ultralytics import YOLO
from yolo_to_vgg import yolo_results_to_vgg
from datetime import datetime

root_dir = Path("big_images/")
# collecte toutes les images (insensible à la casse)
jpg_paths = [str(p) for p in root_dir.rglob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"]]
print(f"{datetime.now()} Total images found: {len(jpg_paths)}", flush=True)

exp_names = {
    'color': ['beige', 'bleu'],
    'typo': ['boucharde', 'flamme', 'spuntato'],
    'state': ['degrade', 'satisfaisant'],
}

bs = 64             # batch size (pour ton affichage)
model_batch = 16    # param 'batch' passé à ultralytics (inference internal batching)
chunk_size = 10000  # taille des sous-chunks
save_every_chunks = True  # si True, on sauve un JSON par chunk pour limiter mémoire

log_file = "predict_run.log"
def log(s):
    print(s, flush=True)
    with open(log_file, "a") as f:
        f.write(f"{datetime.now()} {s}\n")

for exp in exp_names.keys():
    log(f"\n=== Starting experiment: {exp} ===")
    model = YOLO(f"nancy_v2/pave_stanislas_{exp}/weights/best_{exp}.pt", task="segment")
    model_name = f"big_test_{exp}"

    # résultat accumulé (on sauvegarde par chunk pour éviter 38k en RAM)
    results_accum = []
    chunk_idx = 0

    for chunk_start in range(0, len(jpg_paths), chunk_size):
        chunk_paths = jpg_paths[chunk_start: chunk_start + chunk_size]
        log(f"[{exp}] Processing chunk {chunk_idx}: {chunk_start} -> {chunk_start + len(chunk_paths) - 1} (size={len(chunk_paths)})")
        batch_idx = 0

        for nb in range(0, len(chunk_paths), bs):
            batch_paths = chunk_paths[nb: nb + bs]
            log(f"[{exp}]   batch {batch_idx} -> {batch_idx + len(batch_paths) - 1}",)
            try:
                # Tentative 1: utiliser stream=True (contourne certaines limites internes)
                try:
                    batch_results_gen = model(
                        batch_paths,
                        imgsz=512,
                        save=True,
                        retina_masks=True,
                        agnostic_nms=True,
                        show_conf=False,
                        batch=model_batch,
                        conf=.25,
                        iou=.5,
                        verbose=True,
                        device=0,
                        stream=True,   # essaie le mode stream (générateur)
                    )
                    # si aucun generateur mais une liste est retournée, on lève pas d'erreur
                    if hasattr(batch_results_gen, "__iter__") and not isinstance(batch_results_gen, list):
                        batch_results = list(batch_results_gen)  # consommer le generator
                    else:
                        batch_results = batch_results_gen  # peut déjà être une liste
                except TypeError:
                    # fallback si la version d'ultralytics n'accepte pas stream
                    batch_results = model(
                        batch_paths,
                        imgsz=512,
                        save=True,
                        retina_masks=True,
                        agnostic_nms=True,
                        show_conf=False,
                        batch=model_batch,
                        conf=.25,
                        iou=.5,
                        verbose=True,
                        device=0,
                    )

                # sécurité si None
                if batch_results is None:
                    log(f"[{exp}]     WARNING: model returned None for batch {batch_idx}. Skipping.")
                    batch_idx += 1
                    # nettoyage et continuer
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue

                # forcer sur cpu pour économiser GPU ram
                batch_results = [r.cpu() for r in batch_results]
                results_accum.extend(batch_results)

                # nettoyage après chaque batch
                del batch_results
                gc.collect()
                if torch.cuda.is_available():
                    try: torch.cuda.synchronize()
                    except Exception: pass
                    torch.cuda.empty_cache()
                    try: torch.cuda.ipc_collect()
                    except Exception: pass

            except Exception as e:
                # On log l'erreur ET on continue (ne casse pas tout)
                log(f"[{exp}]   ERROR on batch {batch_idx} (global {chunk_start + nb}): {repr(e)}")
                # sauvegarde partielle des résultats accumulés pour sécurité
                try:
                    if results_accum:
                        part_name = f"{model_name}_chunk{chunk_idx}_partial.json"
                        yolo_results_to_vgg(results_accum, model.names, part_name, offset=128)
                        log(f"[{exp}]   Saved partial results to {part_name} after error.")
                        results_accum = []
                except Exception as e2:
                    log(f"[{exp}]   ERROR while saving partial results: {repr(e2)}")
                # continue to next batch
            batch_idx += 1

        # fin du chunk -> sauvegarde des résultats du chunk
        try:
            if results_accum:
                out_name = f"{model_name}_chunk{chunk_idx}.json" if save_every_chunks else f"{model_name}.json"
                # si on veut un seul fichier final, on pourrait append; ici on sauvegarde par chunk
                yolo_results_to_vgg(results_accum, model.names, out_name, offset=128)
                log(f"[{exp}]   Saved chunk results to {out_name} (items={len(results_accum)})")
                # libérer la liste si on sauve par chunk
                if save_every_chunks:
                    results_accum = []
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        except Exception as e:
            log(f"[{exp}]   ERROR saving chunk {chunk_idx}: {repr(e)}")

        chunk_idx += 1

    # cleanup de fin d'expérience
    try:
        del results_accum, model
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try: torch.cuda.ipc_collect()
        except Exception: pass

    log(f"=== Finished experiment: {exp} ===")
