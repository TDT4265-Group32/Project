import imageio.v2 as imageio
import os
import cv2
import argparse
import glob
from tqdm import tqdm
from moviepy.editor import VideoFileClip

def create_video(src_path: str,
                 filename: str = "unnamed_video",
                 extension: str = 'gif',
                 fps: int = 5):
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

    match extension:
        case 'gif':
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        case 'mp4':
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        case _:
            raise ValueError(f'Unsupported extension: {extension}')

    video = cv2.VideoWriter(f'{filename}.mp4', fourcc, fps, (width, height))

    try:
        for image in tqdm(images, desc=f'Creating video'):
            video.write(cv2.imread(os.path.join(src_path, image)))
    except Exception as e:
        raise Exception(f'Encountered error: {e} while creating video.')

    video.release()
    print(f'Video created at: {filename}.mp4')
    
    if extension == 'gif':
        clip = VideoFileClip(f'{filename}.mp4')
        clip.write_gif(f'{filename}.gif')
        print(f'GIF created at: {filename}.gif')
