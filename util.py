import imageio
from PIL import Image
import os

# Print out the number of frames in the GIF
def get_num_frames(input_gif):
    reader = imageio.get_reader(input_gif)
    num_frames = len(reader)
    return num_frames

def create_images_from_gif(input_gif, image_path_folder):
    if not os.path.exists(image_path_folder):
        os.makedirs(image_path_folder)

    if not os.path.isdir(image_path_folder):
        raise NotADirectoryError(f"{image_path_folder} is not a valid directory")

    reader = imageio.get_reader(input_gif)
    for i, frame in enumerate(reader):
        image = Image.fromarray(frame)
        image.save(os.path.join(image_path_folder, f"frame_{i}.png"))

def create_gif_from_images(image_folder, output_gif_path, duration=500):
    images = [Image.open(image_path) for image_path in get_image_paths_in_folder(image_folder)]
    images[0].save(
        output_gif_path,
        save_all=True,
        append_images=images[1:],
        optimize=False,
        duration=duration,
        loop=0  # 0 means infinite loop
    )

# Function that returns a list of the image paths in a folder
def get_image_paths_in_folder(folder_path):
    image_paths = sorted(
        [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg', '.gif'))],
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[-1])
    )
    return image_paths

def change_gif_framerate(input_gif, output_gif, duration):
    create_images_from_gif(input_gif, "frames")
    create_gif_from_images("frames", output_gif, duration)

if __name__ == "__main__":
    change_gif_framerate("output.gif", "output.gif", 100)