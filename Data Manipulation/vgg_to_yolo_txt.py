import json
import cv2
import os

# Define the class mapping

with open('/home/adelb/Documents/Bpartners/revets/VGG/big_vgg2.json') as f:
    data = json.load(f)
    
classes = sorted(set(reg["region_attributes"]['label'] for file in data.values() for reg in file['regions'].values()))

class_mapping = {i: cls for i, cls in enumerate(classes)}
print(class_mapping)

# Create a reverse mapping to get class numbers by label
label_to_class_number = {v: k for k, v in class_mapping.items()}

def read_json(json_file):
    """Read data from JSON file."""
    with open(json_file) as f:
        json_obj = json.load(f)
    return json_obj

def get_img_names(json_obj):
    """Get image names from JSON - top level in hierarchy."""
    return list(json_obj.keys())

def get_img_shape(img_file):
    """Get image shape."""
    img = cv2.imread(img_file)
    if img is None:
        raise ValueError(f"Image not found or unable to read: {img_file}")
    return img.shape  # (height, width, channels)

def calc_relative_points(points_x, points_y, img_shape):
    """Convert polygon points to relative coordinates."""
    img_height, img_width = img_shape[:2]
    rel_points = [(min(round(x / img_width, 3), 0.99), min(round(y / img_height, 3), 0.99)) for x, y in zip(points_x, points_y)]
    return rel_points


def write_yolo_txt(polygons, img_class, img_filename, output_dir):
    """Write polygon annotations to a txt file in YOLO format."""
    txt_filename = os.path.join(output_dir, img_filename.rsplit('.', 1)[0] + ".txt")
    num_annots = len(img_class)
    
    with open(txt_filename, "w") as f:
        for i in range(num_annots):
            # Map the class label to the corresponding class number
            class_number = label_to_class_number.get(img_class[i], -1)
            if class_number == -1:
                print(f"Erreur : Label '{img_class[i]}' non trouvé dans le mapping.")
                continue
            
            # Compose a row with the class number followed by polygon points
            polygon_str = " ".join([f"{x} {y}" for x, y in polygons[i]])
            row = f"{class_number} {polygon_str}\n"
            f.write(row)

def convert_to_yolo(json_file, img_dir, output_dir):
    """Convert JSON annotations to YOLO format and write outputs to a specified directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    json_obj = read_json(json_file)
    img_names = get_img_names(json_obj)
    
    for img in img_names:
        img_class = []
        polygons = []

        # Loop over regions in JSON (for a given image) to get polygon coordinates
        for key, region in json_obj[img]['regions'].items():
            if isinstance(region, dict) and 'region_attributes' in region:
                region_attributes = region['region_attributes']
                if isinstance(region_attributes, dict) and 'label' in region_attributes:
                    img_class.append(region_attributes['label'])
                else:
                    print(f"Erreur : 'region_attributes' n'a pas de 'label' ou n'est pas un dictionnaire pour l'image {img}.")
                    continue  # Passer cette région si elle n'est pas valide

                # Extract polygon points
                shape_attributes = region['shape_attributes']
                points_x = shape_attributes['all_points_x']
                points_y = shape_attributes['all_points_y']

                # Convert to relative coordinates
                img_shape = get_img_shape(img_dir + img)
                rel_points = calc_relative_points(points_x, points_y, img_shape)
                polygons.append(rel_points)

            else:
                print(f"Erreur : Région non valide pour l'image {img}.")
                continue

        # Write polygon annotations in YOLO format
        write_yolo_txt(polygons, img_class, img, output_dir)
    
    return 0

# Set the paths and run the function
img_dir = "/home/adelb/Documents/Bpartners/revets/dataset1/images/"
json_file = "/home/adelb/Documents/Bpartners/revets/VGG/big_vgg.json"
output_dir = "/home/adelb/Documents/Bpartners/revets/dataset1/labels"
convert_to_yolo(json_file, img_dir, output_dir)
