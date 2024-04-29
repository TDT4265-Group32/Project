import os
import shutil

def copy_datasets():
    """Copy the NAPLab-LiDAR dataset from the source directory to the target directory.
    
    The source directory is determined by checking if the directory exists on Cybele and Idun.
    """
    cybele_dir = os.path.expanduser("~/../../datasets/tdt4265/ad/NAPLab-LiDAR/")
    idun_dir = os.path.expanduser("~/../../cluster/projects/vc/data/ad/open/NAPLab-LiDAR")
    target_dir = "datasets/NAPLab-LiDAR/"
    
    if os.path.exists(os.path.dirname(target_dir)):
        print("Dataset already exists.")
        return

    # Check if source directory exists
    if os.path.exists(os.path.dirname(cybele_dir)):
        source_dir = cybele_dir
    elif os.path.exists(os.path.dirname(idun_dir)):
        source_dir = idun_dir
    else:
        raise FileNotFoundError("Dataset not found on Cybele or Idun. Please check the dataset location and try again.")

    # Copy all files from source directory to target directory
    for file_name in os.listdir(source_dir):
        full_file_name = os.path.join(source_dir, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, target_dir)

if __name__ == "__main__":
    copy_datasets()