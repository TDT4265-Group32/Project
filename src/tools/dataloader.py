import os
import shutil
import glob
from tqdm import tqdm

def extract_dataset():
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
            print("Dataset already exists in the target directory, skipping extraction.")
            return

    # Check if images and labels already exist in the target directory
    image_dir = os.path.join(target_dir, "images")
    label_dir = os.path.join(target_dir, "labels")
    images_exist = all(os.path.exists(os.path.join(image_dir, f"frame_{i:06d}.PNG")) for i in range(1905))
    labels_exist = all(os.path.exists(os.path.join(label_dir, f"frame_{i:06d}.txt")) for i in range(1905))

    if images_exist and labels_exist:
        print("Images and labels already exist in the target directory.")
        return
    else:
        # Copy all files from source directory to target directory
        print(f"Copying dataset from '{source_dir}' to '{target_dir}'...")
        shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)
        print("Dataset copied successfully.")

def export_data(architecture: str):
    
    # Create the directory if it doesn't exist
    if not os.path.exists('export'):
        os.makedirs('export')
    else:
        shutil.rmtree('export')
        os.makedirs('export')

    # Find most recent training folder
    training_folder = sorted(glob.glob('runs/detect/train*'))[-1]
    validation_folder = sorted(glob.glob('runs/detect/val*'))[-1]
    prediction_folder = sorted(glob.glob(f'results/{architecture}'))[-1]
    configs_folder = sorted(glob.glob(f'configs/{architecture}'))[-1]
    emissions_file = glob.glob('emissions.csv')[-1]

    print('Attempting to export data...')
    try:
        shutil.copytree(training_folder, 'export/train')
        shutil.copytree(validation_folder, 'export/val')
        shutil.copytree(prediction_folder, 'export/predict')
        shutil.copytree(configs_folder, 'export/configs')
        shutil.copy(emissions_file, 'export/emissions.csv')
    except FileNotFoundError as e:
        print(f'{e}: Please ensure that the training, validation, and prediction folders exist.')
    else:
        print('Data exported successfully.')

if __name__ == "__main__":
    print("Copying dataset...")