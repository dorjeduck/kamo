import os
import re
import numpy as np

import imageio.v2 as imageio  # Import imageio.v2 to avoid deprecation warning
from PIL import Image

def create_gif(image_folder, gif_name, total_duration, pause_duration):
    # Helper function to extract the numeric part of the filename
    def extract_number(filename):
        match = re.findall(r'\d+', filename)
        return int(match[-1]) if match else 0

    # Get all image file paths from the folder
    images = [os.path.join(image_folder, file) for file in os.listdir(image_folder) if file.endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif'))]
    
    # Sort the images by numeric value in the filename
    images.sort(key=lambda x: extract_number(os.path.basename(x)))
    
    # Load and resize images
    frames = []
    for image_path in images:
        img = Image.open(image_path)
        img_resized = img.resize((img.width // 2, img.height // 2))  # Resize to half the size
        frames.append(img_resized)
    
    # Calculate the duration per frame
    num_images = len(images)

    if num_images == 0:
        print("nothing to do")
        return
    duration_per_frame = total_duration / num_images
    duration_per_frame_ms = int(duration_per_frame * 1000)  # Convert to milliseconds

    # Convert resized frames to a list of imageio images with explicit durations
    imageio_frames = [(np.array(img), duration_per_frame_ms) for img in frames]

    # Add pause frames
    pause_frame = np.array(frames[-1])  # Use the last frame for the pause
    pause_frames = [(pause_frame, duration_per_frame_ms)] * int(pause_duration / duration_per_frame)

    # Combine frames and pause frames
    final_frames = imageio_frames + pause_frames

    # Create the GIF with specified duration per frame and pause before restart
    imageio.mimsave(gif_name, [frame[0] for frame in final_frames], duration=[frame[1] for frame in final_frames], loop=0)
    print(f"GIF created successfully and saved as {gif_name}")

# Example usage
image_folder = 'imgs'  # Specify your image folder path
gif_name = 'output.gif'  # Specify the name of the output GIF file
total_duration = 10.0  # Total duration of the GIF in seconds (excluding pause)
pause_duration = 5.0  # Pause duration before the GIF restarts in seconds

create_gif(image_folder, gif_name, total_duration, pause_duration)
