import os
import argparse
import json
from PIL import Image,ImageDraw
import numpy as np
from xml.etree import ElementTree as ET
import cv2

def crop_and_save_object(image_path, segmentation, save_img_dir, category_id, image_id, obj_id):
    # Load the image
    image = Image.open(image_path).convert("RGBA")  # Ensure image is in RGBA format
    image_np = np.array(image)

    # Create a mask based on the segmentation points
    mask = np.zeros(image_np.shape[:2], dtype=np.uint8)  # Create an empty mask

    # Fill the mask using the segmentation points
    if len(segmentation) % 2 == 0:  # Ensure it's a valid polygon
        polygon = np.array(segmentation).reshape((-1, 1, 2)).astype(np.int32)
        cv2.fillPoly(mask, [polygon], color=1)  # Fill the polygon in the mask

    # Ensure mask is 3D
    mask_3d = mask[:, :, np.newaxis]  # Add channel dimension

    # Create the cropped image using the mask
    cropped_image_np = image_np * mask_3d  # This will broadcast correctly

    # Extract the bounding box coordinates
    y_indices, x_indices = np.where(mask == 1)  # Find the indices of the mask
    if len(y_indices) > 0 and len(x_indices) > 0:
        y_min, y_max = y_indices.min(), y_indices.max()
        x_min, x_max = x_indices.min(), x_indices.max()
        cropped_image_np = cropped_image_np[y_min:y_max+1, x_min:x_max+1]

        # Save the cropped image
        obj_filename = os.path.join(save_img_dir, f"{image_id}_obj{obj_id}.png")
        Image.fromarray(cropped_image_np).save(obj_filename)

        # Return the filename and new bounding box
        new_bbox = [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1]  # Format: [xmin, ymin, width, height]
        new_polygon = segmentation  # You can modify this if necessary
        return obj_filename, new_bbox, new_polygon
    else:
        print(f"No valid mask found for object {obj_id} in image {image_id}.")
        return None, None, None


def process_yolo(annotation_dir, image_dir, output_dir):
    save_img_dir = os.path.join(output_dir, "Cropped_images")
    save_lbl_dir = os.path.join(output_dir, "Cropped_labels")

    os.makedirs(save_img_dir, exist_ok=True)
    os.makedirs(save_lbl_dir, exist_ok=True)

    images = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    for image_file in images:
        image_id = os.path.splitext(image_file)[0]
        image_path = os.path.join(image_dir, image_file)
        annotation_file = os.path.join(annotation_dir, f"{image_id}.txt")

        if os.path.exists(annotation_file):
            with open(annotation_file, 'r') as file:
                lines = file.readlines()

            for obj_id, line in enumerate(lines):
                components = line.split()
                class_id = int(components[0])
                polygon_points = list(map(float, components[1:]))

                # Crop and save the object
                obj_filename, new_bbox, new_polygon = crop_and_save_object(
                    image_path, polygon_points, save_img_dir, class_id, image_id, obj_id)

                if obj_filename is not None:
                    new_label_file_name = os.path.splitext(os.path.basename(obj_filename))[0] + ".txt"
                    new_label_file_path = os.path.join(save_lbl_dir, new_label_file_name)

                    # Write new annotation for the cropped image
                    with open(new_label_file_path, 'w') as label_file:
                        # YOLO expects the format: class_id followed by normalized bounding box or polygon points
                        label_file.write(f"{class_id} {' '.join(map(str, new_polygon))}\n")


def process_voc(annotation_dir, image_dir, output_dir):
    save_img_dir = os.path.join(output_dir, "Cropped_images")
    save_lbl_dir = os.path.join(output_dir, "Cropped_labels")

    os.makedirs(save_img_dir, exist_ok=True)
    os.makedirs(save_lbl_dir, exist_ok=True)

    images = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    for image_file in images:
        image_id = os.path.splitext(image_file)[0]
        image_path = os.path.join(image_dir, image_file)
        annotation_file = os.path.join(annotation_dir, f"{image_id}.xml")

        if os.path.exists(annotation_file):
            tree = ET.parse(annotation_file)
            root = tree.getroot()

            for obj_id, obj in enumerate(root.findall('object')):
                obj_class = obj.find('name').text

                # Extract the segmentation polygon points if available
                polygon = obj.find('segmentation')
                if polygon is not None:
                    # Replace commas with spaces, then split and convert to float
                    polygon_points = list(map(float, polygon.text.strip().replace(',', ' ').split()))
                else:
                    # Use bounding box as a fallback if no polygon is available
                    bbox = obj.find('bndbox')
                    xmin = int(bbox.find('xmin').text)
                    ymin = int(bbox.find('ymin').text)
                    xmax = int(bbox.find('xmax').text)
                    ymax = int(bbox.find('ymax').text)
                    polygon_points = [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]  # convert bbox to polygon

                # Crop and save the object
                obj_filename, new_bbox, new_polygon = crop_and_save_object(
                    image_path, polygon_points, save_img_dir, obj_class, image_id, obj_id)

                if obj_filename is not None:
                    new_annotation_file = os.path.join(save_lbl_dir, f"{image_id}_obj{obj_id}.xml")

                    # Update bounding box for the cropped image
                    bbox = ET.SubElement(obj, 'bndbox')  # Create new bndbox if necessary
                    xmin_elem = ET.SubElement(bbox, 'xmin')
                    ymin_elem = ET.SubElement(bbox, 'ymin')
                    xmax_elem = ET.SubElement(bbox, 'xmax')
                    ymax_elem = ET.SubElement(bbox, 'ymax')

                    xmin_elem.text = str(new_bbox[0])
                    ymin_elem.text = str(new_bbox[1])
                    xmax_elem.text = str(new_bbox[0] + new_bbox[2])  # x_min + width
                    ymax_elem.text = str(new_bbox[1] + new_bbox[3])  # y_min + height

                    # Ensure the polygon points are updated correctly for the new cropped image
                    if polygon is not None:
                        # Update the segmentation points for the new cropped image
                        polygon.text = ' '.join(map(str, new_polygon))
                    else:
                        # If no polygon was present, create one from the new bounding box
                        polygon = ET.SubElement(obj, 'segmentation')
                        polygon.text = ' '.join(map(str, new_polygon))

                    # Save the updated annotation to a new XML file
                    tree.write(new_annotation_file)

                    # Optional: Remove the original object if needed
                    # root.remove(obj)

 
def process_coco(annotation_path, image_dir, output_dir):
    save_img_dir = os.path.join(output_dir, "Cropped_images")
    os.makedirs(save_img_dir, exist_ok=True)

    # Load the COCO data and initialize new data structure
    with open(annotation_path, 'r') as f:
        coco_data = json.load(f)

    new_coco_data = {
        "images": [],
        "annotations": [],
        "categories": coco_data['categories']  # Retain the original categories
    }

    # Process each image in the COCO data
    for image_info in coco_data['images']:
        image_id = int(image_info['id'])
        image_filename = image_info['file_name']
        image_path = os.path.join(image_dir, image_filename)

        annotations = [ann for ann in coco_data['annotations'] if int(ann['image_id']) == image_id]

        for obj_id, ann in enumerate(annotations):
            category_id = int(ann['category_id'])
            
            # Check if the annotation has a segmentation field and if it's a polygon list
            if 'segmentation' in ann and isinstance(ann['segmentation'], list) and len(ann['segmentation']) > 0:
                segmentation = ann['segmentation'][0]  # Assuming we're using the first polygon
            else:
                print(f"Skipping annotation {obj_id} in image {image_filename} due to missing or invalid segmentation.")
                continue  # Skip this object if there's no valid segmentation
            
            # Crop and save the object using the segmentation polygon points
            obj_filename, new_bbox, new_polygon = crop_and_save_object(
                image_path, segmentation, save_img_dir, category_id, image_id, obj_id)

            if obj_filename is not None:
                # Add the cropped image data to the new COCO data structure
                new_coco_data['images'].append({
                    "id": int(obj_id),
                    "file_name": obj_filename,
                    "width": int(new_bbox[2]),
                    "height": int(new_bbox[3])
                })

                # Ensure that the bbox and segmentation points are standard Python types
                new_bbox = [int(x) if isinstance(x, np.integer) else x for x in new_bbox]
                new_polygon = [float(x) if isinstance(x, np.floating) or isinstance(x, np.integer) else x for x in new_polygon]

                # Add the new annotation for the cropped image
                new_coco_data['annotations'].append({
                    "image_id": int(obj_id),  # Link to cropped image ID
                    "category_id": int(category_id),
                    "segmentation": [new_polygon],
                    "bbox": new_bbox,
                    "id": int(obj_id)
                })

    # Save all annotations into a single COCO-format JSON file
    output_annotation_file = os.path.join(output_dir, "coco_annotations.json")
    with open(output_annotation_file, 'w') as f:
        json.dump(new_coco_data, f, indent=4)

def process_dataset(image_dir, annotation_path, annotation_type, output_dir):
    if annotation_type == "COCO":
        process_coco(annotation_path, image_dir, output_dir)
    elif annotation_type == "VOC":
        process_voc(annotation_path, image_dir, output_dir)
    elif annotation_type == "YOLO":
        process_yolo(annotation_path, image_dir, output_dir)
    else:
        raise ValueError(f"Unsupported annotation type: {annotation_type}")


if __name__ == "__main__":
    dir = os.getcwd()
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image_dir", required=True, help="Path to the image directory")
    ap.add_argument("-a", "--annotation_path", required=True, help="Path to the annotation directory or file")
    ap.add_argument("-o", "--output_dir", default=os.path.join(dir, "Cropped_Outputs/"), help="Path to the output directory")
    args = vars(ap.parse_args())
    
    image_dir = args["image_dir"]
    annotation_path = args["annotation_path"]
    output_dir = args["output_dir"]

    os.makedirs(output_dir, exist_ok=True)

    if os.path.isfile(annotation_path):
        annotation_type = "COCO"
    elif os.path.isdir(annotation_path):
        files = os.listdir(annotation_path)
        if files[0].endswith(".txt"):
            annotation_type = "YOLO"
        elif files[0].endswith(".xml"):
            annotation_type = "VOC"
    else:
        raise ValueError(f"Unknown annotation format: {annotation_path}")
    
    process_dataset(image_dir, annotation_path, annotation_type, output_dir)