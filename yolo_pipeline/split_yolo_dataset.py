from random import shuffle
import shutil, os

DATA_DIR= '../plantes-invasives/new-yolo-dataset'
IMAGES_DIR= "../plantes-invasives/new-dataset/images"
LABELS_DIR= "../plantes-invasives/new-dataset/labels"

# train: 80%, valid: 15%, test: 5%
TRAIN_RATIO= 0.80 
VALID_RATIO= 0.95

os.makedirs(f"{DATA_DIR}/train/images", exist_ok=True)
os.makedirs(f"{DATA_DIR}/train/labels", exist_ok=True)

os.makedirs(f"{DATA_DIR}/valid/images", exist_ok=True)
os.makedirs(f"{DATA_DIR}/valid/labels", exist_ok=True)

os.makedirs(f"{DATA_DIR}/test/images", exist_ok=True)
os.makedirs(f"{DATA_DIR}/test/labels", exist_ok=True)

x_train_dir = os.path.join(DATA_DIR, "train", "images")
y_train_dir = os.path.join(DATA_DIR, "train", "labels")

x_valid_dir = os.path.join(DATA_DIR, "valid", "images")
y_valid_dir = os.path.join(DATA_DIR, "valid", "labels")

x_test_dir = os.path.join(DATA_DIR, "test", "images")
y_test_dir = os.path.join(DATA_DIR, "test", "labels")

list_images= os.listdir(LABELS_DIR)

print(len(list_images))

shuffle(list_images)
shuffle(list_images)

split_index_1 = int(len(list_images) * TRAIN_RATIO)
split_index_2 = int(len(list_images) * VALID_RATIO)

train_images = list_images[:split_index_1]
valid_images = list_images[split_index_1: split_index_2]
test_images = list_images[split_index_2:]

for fn in train_images:
    if not fn.endswith('.txt') : continue
    shutil.copy(f"{LABELS_DIR}/{fn}", f"{y_train_dir}/{fn}")
    shutil.copy(f"{IMAGES_DIR}/{fn.replace('.txt', '.jpg')}", f"{x_train_dir}/{fn.replace('.txt', '.jpg')}")

for fn in valid_images:
    if not fn.endswith('.txt') : continue
    shutil.copy(f"{LABELS_DIR}/{fn}", f"{y_valid_dir}/{fn}")
    shutil.copy(f"{IMAGES_DIR}/{fn.replace('.txt', '.jpg')}", f"{x_valid_dir}/{fn.replace('.txt', '.jpg')}")

for fn in test_images:
    if not fn.endswith('.txt') : continue
    shutil.copy(f"{LABELS_DIR}/{fn}", f"{y_test_dir}/{fn}")
    shutil.copy(f"{IMAGES_DIR}/{fn.replace('.txt', '.jpg')}", f"{x_test_dir}/{fn.replace('.txt', '.jpg')}")