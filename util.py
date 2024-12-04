import imageio
import random
from PIL import Image
import os

random.seed(24)

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
    """
    ## Params
    `duration`: time duration of each frame, in milliseconds
    """
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
    
    
def partition_list(lst, num_components):
    """Randomly split the list into subsets of some minimum size."""
    
    if len(lst) < num_components:
        return [lst] * num_components # should return the initial list num_components times 
    else:
        # use partition_int to split the list into sizes, and then take those sizes after shuffling
        random.shuffle(lst)
        partitions = []
        sizes = partition_int(len(lst), num_components)
        for size in sizes:
            partitions.append(lst[:size])
            lst = lst[size:]
        return partitions

def partition_int(n, k):
    if k == 1:
        return [n]
    splits = sorted(random.sample(range(1, n), k - 1))
    sizes = [splits[0]] + [splits[i] - splits[i - 1] for i in range(1, k - 1)] + [n - splits[-1]]
    return sizes 


def sanitize_keys(data):
    
    if not isinstance(data, dict):
        raise TypeError("Input must be a dictionary.")

    sanitized_data = {}
    for key, value in data.items():
        if not isinstance(key, (str, int, float, bool, type(None))):
            key = str(key) 
        if isinstance(value, dict):
            value = sanitize_keys(value)

        sanitized_data[key] = value

    return sanitized_data

def compute_mean_and_std(data):
    """Compute the mean and standard deviation of a list of numbers."""
    n = len(data)
    mean = sum(data) / n
    std = (sum((x - mean) ** 2 for x in data) / n) ** 0.5
    return mean, std


if __name__ == "__main__":
    with open('remaining_vehicles.txt', 'r') as file:
        data = [list(map(int, line.strip().split())) for line in file.readlines()]

    mean, std = compute_mean_and_std(data[0])
    print(f"Mean: {mean}, Standard Deviation: {std}")