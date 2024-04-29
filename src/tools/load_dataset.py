import os
import shutil
from tqdm import tqdm

def copy_datasets():
    """Copy the NAPLab-LiDAR dataset from the source directory to the target directory.
    
    The source directory is determined by checking if the directory exists on Cybele and Idun.
    """
    cybele_dir = os.path.expanduser("~/../../datasets/tdt4265/ad/NAPLab-LiDAR/")
    idun_dir = os.path.expanduser("~/../../cluster/projects/vc/data/ad/open/NAPLab-LiDAR")
    target_dir = "datasets/NAPLab-LiDAR/"

    # Check if source directory exists
    if os.path.exists(os.path.dirname(cybele_dir)):
        source_dir = cybele_dir
    elif os.path.exists(os.path.dirname(idun_dir)):
        source_dir = idun_dir
    else:
        if not os.path.exists(target_dir):
            raise FileNotFoundError("Dataset not found on Cybele or Idun. Please check the dataset location and try again.")
        else:
            print("Dataset already exists in the target directory.")
            return

    # Copy all files from source directory to target directory
    print(f"Copying dataset from '{source_dir}' to '{target_dir}'...")
    shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)
    print("Dataset copied successfully.")
    
    test_img_dir = os.path.join(target_dir, "images", "test")
    test_lbl_dir = os.path.join(target_dir, "labels", "test")
    os.makedirs(test_img_dir, exist_ok=True)
    os.makedirs(test_lbl_dir, exist_ok=True)
    
    image_dir = os.path.join(target_dir, "images")
    label_dir = os.path.join(target_dir, "labels")
    # Move frames 201-301 to the test directory
    for i in tqdm(range(201, 302), desc="Moving test images and labels"):
        image_name = f"frame_{i:06d}.PNG"
        label_name = f"frame_{i:06d}.txt"
        source_image_path = os.path.join(image_dir, image_name)
        source_label_path = os.path.join(label_dir, label_name)
        try:
            destination_image_path = os.path.join(test_img_dir, image_name)
            destination_label_path = os.path.join(test_lbl_dir, label_name)
            shutil.move(source_image_path, destination_image_path)
            shutil.move(source_label_path, destination_label_path)
        except FileNotFoundError:
            print(f"File {image_name} or {label_name} not found.")
            continue
    print("Test images and labels moved successfully.")

if __name__ == "__main__":
    copy_datasets()