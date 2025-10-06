from random import shuffle
import shutil, os

DATA_DIR= '/home/adelb/Documents/Bpartners/Pleiades/dataset/bati'
IMAGES_DIR= "/home/adelb/Documents/Bpartners/Pleiades/dataset/bati_2014_cherbourg/images"
MASKS_DIR= "/home/adelb/Documents/Bpartners/Pleiades/dataset/bati_2014_cherbourg/masks"

# train: 80%, valid: 15%, test: 5%
TRAIN_RATIO= 0.80
VALID_RATIO= 0.95

os.makedirs(f"{DATA_DIR}/train/images", exist_ok=True)
os.makedirs(f"{DATA_DIR}/train/masks", exist_ok=True)

os.makedirs(f"{DATA_DIR}/valid/images", exist_ok=True)
os.makedirs(f"{DATA_DIR}/valid/masks", exist_ok=True)

os.makedirs(f"{DATA_DIR}/test/images", exist_ok=True)
os.makedirs(f"{DATA_DIR}/test/masks", exist_ok=True)

x_train_dir = os.path.join(DATA_DIR, "train", "images")
y_train_dir = os.path.join(DATA_DIR, "train", "masks")

x_valid_dir = os.path.join(DATA_DIR, "valid", "images")
y_valid_dir = os.path.join(DATA_DIR, "valid", "masks")

x_test_dir = os.path.join(DATA_DIR, "test", "images")
y_test_dir = os.path.join(DATA_DIR, "test", "masks")

list_images= [f  for f in os.listdir(MASKS_DIR) if f.endswith('.png')]

shuffle(list_images)
shuffle(list_images)

split_index_1= int(len(list_images) * TRAIN_RATIO)
split_index_2= int(len(list_images) * VALID_RATIO)

train_images= list_images[:split_index_1]
valid_images= list_images[split_index_1: split_index_2]
test_images= list_images[split_index_2:]

for fn in train_images:
    shutil.copy(f"{MASKS_DIR}/{fn}", f"{y_train_dir}/{fn}")
    shutil.copy(f"{IMAGES_DIR}/{fn}", f"{x_train_dir}/{fn}")

for fn in valid_images:
    shutil.copy(f"{MASKS_DIR}/{fn}", f"{y_valid_dir}/{fn}")
    shutil.copy(f"{IMAGES_DIR}/{fn}", f"{x_valid_dir}/{fn}")

for fn in test_images:
    shutil.copy(f"{MASKS_DIR}/{fn}", f"{y_test_dir}/{fn}")
    shutil.copy(f"{IMAGES_DIR}/{fn}", f"{x_test_dir}/{fn}")
