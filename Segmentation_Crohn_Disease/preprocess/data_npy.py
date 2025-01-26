import os
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import SimpleITK as sitk
from skimage import transform
from tqdm import tqdm
import cv2
from prompt_generator import get_centreline_points_from_file, centreline_prompt

# Supported file suffixes
img_name_suffixes = [".nii.gz", ".nii"]
gt_name_suffixes = [".nii.gz", ".nii"]
centerline_name_suffix = ".txt"

image_size = 1024  # Target image size

def find_matching_files(gt_path, nii_path, centreline_path):
    """
    Find files that have matching ground truth, MRI image, and centerline files.
    """
    names = sorted(os.listdir(gt_path))
    valid_files = []
    for name in names:
        base_name = name.split('.')[0]  # Extract base name
        matching_img = any(
            os.path.exists(os.path.join(nii_path, base_name + suffix))
            for suffix in img_name_suffixes
        )
        matching_centreline = os.path.exists(os.path.join(centreline_path, base_name + centerline_name_suffix))
        if matching_img and matching_centreline:
            valid_files.append(name)
    return valid_files

def load_image(nii_path, base_name):
    """
    Load an MRI image or ground truth segmentation file with .nii or .nii.gz format.
    """
    for suffix in img_name_suffixes:
        file_path = os.path.join(nii_path, base_name + suffix)
        if os.path.exists(file_path):
            return sitk.ReadImage(file_path)
    raise FileNotFoundError(f"No matching image found for {base_name}")

def process_files(nii_path, gt_path, centreline_path, npy_path):
    """
    Process files to convert .nii/.nii.gz images and ground truth into .npy format.
    """
    os.makedirs(os.path.join(npy_path, "gts"), exist_ok=True)
    os.makedirs(os.path.join(npy_path, "imgs"), exist_ok=True)
    os.makedirs(os.path.join(npy_path, "centreline"), exist_ok=True)

    # Find matching files
    names = find_matching_files(gt_path, nii_path, centreline_path)
    print(f"Number of valid files: {len(names)}")

    # Create CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # Process each file
    for name in tqdm(names):
        base_name = name.split('.')[0]
        gt_name = name
        centreline_name = base_name + centerline_name_suffix

        # Load MRI image, ground truth, and centerline points
        img_sitk = load_image(nii_path, base_name)
        image_data = sitk.GetArrayFromImage(img_sitk)
        gt_sitk = load_image(gt_path, base_name)
        gt_data_ori = np.uint8(sitk.GetArrayFromImage(gt_sitk))
        centreline_points = get_centreline_points_from_file(os.path.join(centreline_path, centreline_name))

        # Find valid slices with both segmentation and centerline data
        z_index = []
        for i in range(len(gt_data_ori)):
            centreline_prompt_points = centreline_prompt(centreline_points, i)
            if np.any(gt_data_ori[i] > 0) and len(centreline_prompt_points) > 0:
                z_index.append(i)

        if len(z_index) > 0:
            for i in z_index:
                # Get the current slice
                img_i = image_data[i, :, :]
                if not np.any(img_i):  # Skip empty slices
                    continue

                # Normalize and apply CLAHE
                img_i = (img_i - img_i.min()) / (img_i.max() - img_i.min()) * 255  # Normalize to 0-255
                img_i = img_i.astype(np.uint8)  # Convert to uint8
                enhanced_img = clahe.apply(img_i)  # Apply CLAHE

                # Resize the enhanced single-channel image
                resized_img = transform.resize(
                    enhanced_img, (image_size, image_size), order=3, preserve_range=True, mode="constant", anti_aliasing=True
                )
                resized_img = np.uint8(resized_img)  # Convert to uint8

                # Resize the ground truth
                gt_i = gt_data_ori[i, :, :]
                resized_gt = transform.resize(
                    gt_i, (image_size, image_size), order=0, preserve_range=True, mode="constant", anti_aliasing=False
                )
                resized_gt = np.uint8(resized_gt)

                # Stack the processed single-channel image into three channels
                img_3c = np.stack([resized_img, resized_img, resized_img], axis=-1)

                # Scale centerline points
                scale_h, scale_w = image_size / gt_data_ori.shape[1], image_size / gt_data_ori.shape[2]
                centreline_points_scaled = centreline_points.copy()
                centreline_points_scaled['x'] *= scale_h
                centreline_points_scaled['y'] *= scale_w
                centreline_points_roi = centreline_points_scaled[centreline_points_scaled['z'] == i][['x', 'y']].to_numpy()

                # Save resized data
                np.save(os.path.join(npy_path, "imgs", f"{base_name}-{str(i).zfill(3)}.npy"), img_3c)
                np.save(os.path.join(npy_path, "gts", f"{base_name}-{str(i).zfill(3)}.npy"), resized_gt)
                np.save(os.path.join(npy_path, "centreline", f"{base_name}-{str(i).zfill(3)}.npy"), centreline_points_roi)


# Tkinter GUI
def select_directory(title):
    """
    Open a directory selection dialog.
    """
    return filedialog.askdirectory(title=title)

def run_processing():
    """
    Run the processing workflow based on user inputs.
    """
    global nii_path, gt_path, centreline_path, npy_path

    if not nii_path:
        messagebox.showerror("Error", "MRI Image Directory not selected!")
        return

    if not gt_path:
        messagebox.showerror("Error", "Ground Truth Directory not selected!")
        return

    if not centreline_path:
        messagebox.showerror("Error", "Centerline Directory not selected!")
        return

    if not npy_path:
        messagebox.showerror("Error", "Output Directory not selected!")
        return

    # Confirm and run processing
    try:
        process_files(nii_path, gt_path, centreline_path, npy_path)
        messagebox.showinfo("Success", "Processing completed!")
    except Exception as e:
        messagebox.showerror("Error", f"Processing failed: {str(e)}")

def update_path(label, path):
    label.config(text=path)

def select_nii_path():
    global nii_path
    nii_path = select_directory("Select MRI Image Directory")
    update_path(nii_label, nii_path)

def select_gt_path():
    global gt_path
    gt_path = select_directory("Select Ground Truth Directory")
    update_path(gt_label, gt_path)

def select_centreline_path():
    global centreline_path
    centreline_path = select_directory("Select Centerline Directory")
    update_path(centreline_label, centreline_path)

def select_npy_path():
    global npy_path
    npy_path = select_directory("Select Output Directory")
    update_path(npy_label, npy_path)

# Initialize paths
ii_path = ""
gt_path = ""
centreline_path = ""
npy_path = ""

# Set up the Tkinter application
root = tk.Tk()
root.title("MRI to NPY Converter")

frame = tk.Frame(root)
frame.pack(pady=20, padx=20)

tk.Label(frame, text="Convert MRI Images to .npy Format", font=("Arial", 16)).pack(pady=10)

nii_label = tk.Label(frame, text="MRI Path: Not Selected", font=("Arial", 12))
nii_label.pack(pady=5)
tk.Button(frame, text="Select MRI Path", command=select_nii_path, font=("Arial", 12)).pack(pady=5)

gt_label = tk.Label(frame, text="GT Path: Not Selected", font=("Arial", 12))
gt_label.pack(pady=5)
tk.Button(frame, text="Select GT Path", command=select_gt_path, font=("Arial", 12)).pack(pady=5)

centreline_label = tk.Label(frame, text="Centerline Path: Not Selected", font=("Arial", 12))
centreline_label.pack(pady=5)
tk.Button(frame, text="Select Centerline Path", command=select_centreline_path, font=("Arial", 12)).pack(pady=5)

npy_label = tk.Label(frame, text="Output Path: Not Selected", font=("Arial", 12))
npy_label.pack(pady=5)
tk.Button(frame, text="Select Output Path", command=select_npy_path, font=("Arial", 12)).pack(pady=5)

start_button = tk.Button(frame, text="Start Processing", command=run_processing, font=("Arial", 14))
start_button.pack(pady=20)

root.mainloop()

