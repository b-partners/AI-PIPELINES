from ultralytics import YOLO
import yaml


exp_names = {
    'overall': ['beige_boucharde_degrade', 'beige_boucharde_satisfaisant', 'beige_flamme_degrade', 'beige_flamme_satisfaisant', 'beige_spuntato_degrade', 'beige_spuntato_satisfaisant', 'bleu_boucharde_degrade', 'bleu_boucharde_satisfaisant', 'bleu_flamme_degrade', 'bleu_flamme_satisfaisant', 'bleu_spuntato_degrade', 'bleu_spuntato_satisfaisant'],
    'color': ['beige', 'bleu'],
    'typo': ['boucharde', 'flamme', 'spuntato'],
    'state': ['degrade', 'satisfaisant'],
    'cs': ['beige_degrade', 'beige_satisfaisant', 'bleu_degrade', 'bleu_satisfaisant'],
    'ct': ['beige_boucharde', 'beige_flamme', 'beige_spuntato', 'bleu_boucharde', 'bleu_flamme', 'bleu_spuntato'],
    'ts': ['boucharde_degrade', 'boucharde_satisfaisant', 'flamme_degrade', 'flamme_satisfaisant', 'spuntato_degrade', 'spuntato_satisfaisant']            
        }
for exp in exp_names.keys():
    yml = {
        'train': f'/home/sagemaker-user/stanislas/datasetsv3/{exp}/train',
        'val': f'/home/sagemaker-user/stanislas/datasetsv3/{exp}/val',
        'nc': len(exp_names[exp]),
        'names': exp_names[exp]
          }
    with open('my_data.yaml', 'w') as f:
        yaml.safe_dump(yml, f)

    
    # Load a pretrained YOLOv8 model (or YOLO('yolov8n.pt'), 'yolov8s.pt', etc.)
    model = YOLO("yolo11n-seg.pt")  # choose 'n', 's', 'm', 'l', 'x' depending on your hardware
    
    # Train the model
    model.train(
        data="my_data.yaml",  # path to YAML file
        hsv_h = 0,
        hsv_s = 0,
        hsv_v = 0,
        scale = 0,
        imgsz=256,
        batch=16,
        name=f"pave_stanislas_{exp}",  # output folder name under runs/train/
        project="nancy_v3",        # optional: change where training logs go
        workers=4,                   # adjust if you have more CPU cores
        device=[0, 1, 2, 3],
        # Fine tuning hyper params
        epochs=150,
        patience= 15,
        cos_lr= True,
    )
