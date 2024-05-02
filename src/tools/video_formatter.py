import os
import cv2
from tqdm import tqdm
from moviepy.editor import VideoFileClip

def create_video(src_path: str,
                 dst_path: str,
                 filename: str,
                 extension: str,
                 fps: int = 5):
    """Create a video from a series of PNGs.
    
    Args:
        src_path (str): The path to the folder containing the PNGs.
        dst_path (str): The path to save the video.
        filename (str): The name of the video file.
        extension (str): The extension of the video file (either 'gif' or 'mp4').
        fps (int): The frames per second of the video.
    """
    assert os.path.exists(src_path), f'Path does not exist: {src_path}'
    assert extension in ['gif', 'mp4'], f'Invalid extension: {extension}. Must be either "gif" or "mp4".'

    # Create the destination folder if it does not exist
    os.makedirs(dst_path, exist_ok=True)
    full_filename = os.path.join(dst_path, filename)
    
    images = [img for img in os.listdir(src_path) if img.endswith(".PNG")]

    # Check if any PNGs were found
    assert len(images) > 0, f'No PNGs found in {src_path}'
    print()
    images = sorted(images)

    # Extract the dimensions of the first frame
    try:
        print('Extracting frame dimensions...', end=' ')
        frame = cv2.imread(os.path.join(src_path, images[0]))
    except Exception as e:
        raise Exception(f'Encountered error: {e} while extracting frame dimensions.')
    else:
        print(f'Successfully extracted dimensions: {frame.shape[0:2]}')
        height, width, _ = frame.shape

    # Create the video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(f'{full_filename}.mp4', fourcc, fps, (width, height))

    try:
        for image in tqdm(images, desc=f'Creating video'):
            video.write(cv2.imread(os.path.join(src_path, image)))
    except Exception as e:
        raise Exception(f'Encountered error: {e} while creating video.')

    video.release()
    print(f'Video created at: {full_filename}.mp4')

    # Convert the video to a GIF    
    if extension == 'gif':
        clip = VideoFileClip(f'{full_filename}.mp4')
        clip.write_gif(f'{full_filename}.gif')
        print(f'GIF created at: {full_filename}.gif')
