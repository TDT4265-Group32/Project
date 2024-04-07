import imageio.v2 as imageio
import os

def main():
    # Directory containing images
    folder = "datasets/NAPLab-LiDAR/images"

    # Get file names of the images
    images = [img for img in os.listdir(folder) if img.endswith(".PNG")]

    # Sort the images by name
    images.sort()

    # Create image list
    images = [imageio.imread(folder + '/' + img) for img in images]

    # Save images as a movie
    imageio.mimsave('LiDAR_video.mp4', images, fps=20)  # fps = frames per second
    
if __name__ == "__main__":
    main()