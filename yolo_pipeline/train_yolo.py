from ultralytics import YOLO

# Load a pretrained YOLOv8 model (or YOLO('yolov8n.pt'), 'yolov8s.pt', etc.)
model = YOLO("yolo11n-seg.pt", task='segment')  # choose 'n', 's', 'm', 'l', 'x' depending on your hardware

# Train the model
model.train(
    data="data_roofs.yaml",  # path to YAML file
    imgsz=512,
    batch=256,
    name="new_dataset_nano",  # output folder name under runs/train/
    project="BP-Roofs",        # optional: change where training logs go
    workers=4, # adjust if you have more CPU cores
    device=[0, 1, 2, 3],
    # Fine tuning hyper params
    epochs=300,
    optimizer='AdamW',
    lr0=0.0005,## Réduit le taux d'apprentissage final
    lrf=0.005,  # Ajustement du facteur de réduction de la LR
    momentum=0.937,#influençant l'incorporation des gradients passés dans la mise à jour actuelle.
    weight_decay=0.005,
    patience= 20,
    cos_lr= True,
    cls = 1.5,
    flipud = 0.2,
    dfl= 5,
    box= 10,
    scale = 0.0,
    dropout= 0.1,
    mixup= 0.2,
    copy_paste=0.2,
    hsv_h= 0.01,
    hsv_s= 0.1,
    hsv_v= 0.1,
    mask_ratio=10,
    mosaic=.8,
    
    # single_cls= True
    
)
