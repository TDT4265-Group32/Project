import imageio.v2 as imageio
import os
import cv2
from tqdm import tqdm

def create_video(folder, destination='unnamed_video.mp4'):
    # Directory containing images

    # Get file names of the images
    images = [img for img in os.listdir(folder) if img.endswith(".PNG")]
    frame = cv2.imread(os.path.join(folder, images[0]))
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(destination, fourcc, 20, (width, height))

    for image in tqdm(images, desc=f'Creating video'):
        video.write(cv2.imread(os.path.join(folder, image)))
    
    print(f'Video created at: {destination}')
    
    cv2.destroyAllWindows()
    video.release()