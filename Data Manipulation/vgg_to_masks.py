import json, os
import numpy as np
import cv2, shutil
from shapely.geometry import Polygon, MultiPolygon
from PIL import Image, ImageEnhance
import albumentations as A 




imsize= (512, 512)

vgg_path = "/home/adelb/Documents/Bpartners/Pleiades/dataset/bati_2014_cherbourg/pleiade_2014_cherbourg_bati.json"

with open(vgg_path) as f:
    vgg= json.load(f)

dest_folder = f"/home/adelb/Documents/Bpartners/Pleiades/dataset/bati_2014_cherbourg/masks"
os.makedirs(dest_folder, exist_ok=True)


classes= set([reg['region_attributes']['label'] for file in vgg.values() for reg in file['regions'].values()])
my_classes= sorted(list(classes))
my_classes.insert(0, 'background')

print(my_classes)
cls_to_shade= {cls: i*255 for i, cls in enumerate(my_classes)}

empty_poly = 0
for fn, file in vgg.items():
    mask= np.zeros(shape=imsize, dtype=np.uint8)
    polygons= []
    labels= []
    for reg in file['regions'].values():
        
        label= reg['region_attributes']['label']
        # if label in classes_to_augment and not to_augment:
        #     to_augment = True
        Xs= reg['shape_attributes']['all_points_x']
        Ys= reg['shape_attributes']['all_points_y']
        
        XYs= zip(Xs, Ys)
        P= Polygon(XYs)
        P= P if P.is_valid else P.buffer(0)
        
        if isinstance(P, MultiPolygon):
            for p in P.geoms:
                polygons.append(p)
                labels.append(label)
        elif isinstance(P, Polygon)  :
            polygons.append(P)
            labels.append(label)
        else: continue
        
        
        
    polygons_with_labels= zip(polygons, labels)
    polygons_with_labels_sorted= sorted(polygons_with_labels, key= lambda x: x[0].area, reverse=True)
    # print(polygons_with_labels_sorted)
    i= 0
    for polygon, label in polygons_with_labels_sorted:
        if polygon.is_empty:
            # print('empty poly')
            empty_poly += 1
            continue
        coords= [(int(x), int(y)) for x, y in polygon.exterior.coords if not polygon.is_empty]
        cv2.fillPoly(mask, [np.array([coords]).reshape(-1,1,2)], cls_to_shade[label])
        # print(label)
    
    
    cv2.imwrite(f"{dest_folder}/{fn}", mask)
print(f"nbr empty poly: {empty_poly}")