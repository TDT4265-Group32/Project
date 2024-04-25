import imageio.v2 as imageio
import os
import cv2
import argparse
import glob
from tqdm import tqdm

def create_video(src_path, dst_path='unnamed_video.mp4'):
    """Create a video from a series of PNGs.
    
    Args:
    folder (str): Folder containing PNGs
    destination (str): Destination of the video
    
    Returns:
    None
    
    Raises:
    Exception: If no PNGs are found in the folder
    Exception: If an error is encountered while extracting frame dimensions
    Exception: If an error is encountered while creating video
    """
    img_paths = glob.glob(os.path.join(src_path, '*.PNG'))
    images = [img for img in os.listdir(src_path) if img.endswith(".PNG")]
    
    assert len(images) > 0, f'No PNGs found in {src_path}'
    print()
    images = sorted(images)
    
    try:
        print('Extracting frame dimensions...', end=' ')
        frame = cv2.imread(os.path.join(src_path, images[0]))
    except Exception as e:
        raise Exception(f'Encountered error: {e} while extracting frame dimensions.')
    else:
        print(f'Successfully extracted dimensions: {frame.shape[0:2]}')
        height, width, _ = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(dst_path, fourcc, 20, (width, height))

    try:
        for image in tqdm(images, desc=f'Creating video'):
            video.write(cv2.imread(os.path.join(src_path, image)))
    except Exception as e:
        raise Exception(f'Encountered error: {e} while creating video.')

    video.release()
    print(f'Video created at: {dst_path}')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Turn series of PNGs into a video.')
    parser.add_argument('--folder', type=str, help='Folder containing PNGs')
    parser.add_argument('--destination', type=str, default='unnamed_video.mp4', help='Destination of the video')
    args = parser.parse_args()
    src_path = args.folder
    dst_path = args.destination
    
    create_video(src_path, dst_path)