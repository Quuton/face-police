import os
import config
import cv2
import shutil
import xml.etree.ElementTree as ET
from sklearn import preprocessing


def initialize_files_and_directories():
    print("[INFO] Beginning file initialization and checks")
    if not os.path.exists(config.DATASET_DIR) or not os.path.exists(config.DATASET_IMAGES_DIR) or not os.path.exists(config.DATASET_ANNOTATIONS_DIR):
        print("No Dataset found, please insert into {config.DATASET_DIR}, you could wget from https://www.kaggle.com/api/v1/datasets/download/andrewmvd/face-mask-detection.")
        exit()
    
    for directory in config.INITIALIZE_DIRECTORIES:
        if not os.path.exists(directory):
            os.makedirs(directory)
    print("[INFO] File initilization finished")

def image_preprocess():
    print("[INFO] Beginning image processing")

    for filename in os.listdir(config.DATASET_IMAGES_DIR):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(config.DATASET_IMAGES_DIR, filename)
            image = cv2.imread(img_path)

            # resized_image = cv2.resize(image, (416, 416))
            processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            output_path = os.path.join(config.PROCESS_IMAGES_DIR, filename)
            cv2.imwrite(output_path, processed_image)
    
    print("[INFO] Finished image processing")

def load_annotations():
    annotations = []
    for xml_file in os.listdir(config.DATASET_ANNOTATIONS_DIR):
        tree = ET.parse(os.path.join(config.DATASET_ANNOTATIONS_DIR, xml_file))
        root = tree.getroot()
        file_name = root.find('filename').text
        objects = []
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            objects.append((class_name, xmin, ymin, xmax, ymax))
        annotations.append((file_name, objects))
    return annotations

def convert_to_yolo_format(image_size, bounding_box):
    image_width = image_size[0]
    image_height = image_size[1]
    width_scale = 1.0 / image_width
    height_scale = 1.0 / image_height
    
    # Calculate the center coordinates and dimensions of the bounding box
    center_x = (bounding_box[0] + bounding_box[2]) / 2.0
    center_y = (bounding_box[1] + bounding_box[3]) / 2.0
    box_width = bounding_box[2] - bounding_box[0]
    box_height = bounding_box[3] - bounding_box[1]
    
    # Scale the values to YOLO format
    normalized_x = center_x * width_scale
    normalized_y = center_y * height_scale
    normalized_width = box_width * width_scale
    normalized_height = box_height * height_scale
    
    return (normalized_x, normalized_y, normalized_width, normalized_height)

def save_yolo_files(annotations, image_directory, label_directory, label_encoder):
    print("[INFO] Beginning saving YOLO data")
    for file_name, objects in annotations:
        image_path = os.path.join(config.PROCESS_IMAGES_DIR, file_name)
        image = cv2.imread(image_path)

        image_height, image_width, _ = image.shape
        
        # Copy the image to the specified directory
        shutil.copy(image_path, image_directory)
        
        # Prepare the label file path
        label_file_path = os.path.join(label_directory, file_name.replace('.png', '.txt'))
        
        # Write the annotations to the label file
        with open(label_file_path, 'w') as label_file:
            for obj in objects:
                class_name, xmin, ymin, xmax, ymax = obj
                class_id = label_encoder.transform([class_name])[0]
                
                # Convert bounding box to YOLO format
                yolo_bounding_box = convert_to_yolo_format((image_width, image_height), (xmin, ymin, xmax, ymax))
                
                # Write the class ID and bounding box to the label file
                label_file.write(f"{class_id} {' '.join(map(str, yolo_bounding_box))}\n")

    print("[INFO] Done saving YOLO files")

def create_yaml_config(images_training_directory, images_testing_directory, label_encoder):
    data_yaml = f"""
    train: {config.PROCESS_TRAIN_IMAGES_DIR}
    test: {config.PROCESS_TEST_IMAGES_DIR}
    val: {config.PROCESS_VALIDATE_IMAGES_DIR}

    nc: {len(label_encoder.classes_)}
    names: {label_encoder.classes_.tolist()}
    """

    with open(f"{config.PROCESS_DIR}data.yaml", 'w') as file:
        file.write(data_yaml)

def main():
    initialize_files_and_directories()

    image_preprocess()

    annotations = load_annotations()

    if (config.TRAIN_PROPORTION + config.TEST_PROPORTION + config.VALIDATE_PROPORTION) != 10:
        print("Proportion values do not add up to 10, please check the config file!")
        exit()

    train_annotations = annotations[:int(config.TRAIN_PROPORTION / 10 * len(annotations))]
    test_annotations = annotations[int(config.TEST_PROPORTION / 10 * len(annotations)):]
    validate_annotations = annotations[int(config.VALIDATE_PROPORTION /10 * len(annotations)):]

    label_encoder = preprocessing.LabelEncoder()
    all_labels = []
    
    for annotation in annotations:
        for obj in annotation[1]:
            all_labels.append(obj[0])

    label_encoder.fit(all_labels)

    save_yolo_files(train_annotations, config.PROCESS_TRAIN_IMAGES_DIR, config.PROCESS_TRAIN_LABELS_DIR, label_encoder)
    save_yolo_files(test_annotations, config.PROCESS_TEST_IMAGES_DIR, config.PROCESS_TEST_LABELS_DIR, label_encoder)
    save_yolo_files(test_annotations, config.PROCESS_VALIDATE_IMAGES_DIR, config.PROCESS_VALIDATE_LABELS_DIR, label_encoder)


    create_yaml_config(config.PROCESS_TRAIN_IMAGES_DIR, config.PROCESS_TEST_IMAGES_DIR, label_encoder)

if __name__ == "__main__":
    main()