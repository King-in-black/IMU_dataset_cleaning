import os
import shutil
import numpy as np
import SimpleITK as sitk
from skimage.measure import label
from skimage.morphology import medial_axis
from tkinter import Tk, filedialog, Label, Button, messagebox
from tqdm import tqdm
import re


# ========================== Core Functions ==========================

def overwrite_labels(src_folder, target_folder, OnlyTI=True, OnlyATI=False):
    """
    Overwrite labels in segmentation files and save the modified files to a target folder.

    Args:
    - src_folder (str): Path to the source folder containing segmentation files.
    - target_folder (str): Path to the target folder for saving modified files.
    - OnlyTI (bool): If True, keep only Terminal Ileum (TI) regions.
    - OnlyATI (bool): If True, keep only abnormal TI regions.

    Returns:
    - None
    """
    os.makedirs(target_folder, exist_ok=True)
    for filename in os.listdir(src_folder):
        file_path = os.path.join(src_folder, filename)
        seg = sitk.ReadImage(file_path)
        arr = sitk.GetArrayFromImage(seg)

        # Modify labels based on conditions
        arr = np.where(arr == 6, 0, arr)  # Remove appendix
        arr = np.where(arr == 3, 0, arr)  # Remove colon
        arr = np.where(arr == 4, 0, arr)  # Remove colon
        if OnlyTI:
            arr = np.where(arr == 1, 1, arr)  # Keep abnormal TI
            arr = np.where(arr == 2, 1, arr)  # Keep normal TI
        if OnlyATI:
            arr = np.where(arr == 1, 1, arr)  # Keep abnormal TI
            arr = np.where(arr == 2, 0, arr)  # Remove normal TI

        seg_new = sitk.GetImageFromArray(arr)
        seg_new.CopyInformation(seg)
        sitk.WriteImage(seg_new, os.path.join(target_folder, filename))
    messagebox.showinfo("Done", f"Labels overwritten and saved in {target_folder}")


def tidy_folder(source_folder, target_folder):
    """
    Organize files into 'seg' (segmentation) and 'img' (images) subfolders.

    Args:
    - source_folder (str): Path to the folder containing files to organize.
    - target_folder (str): Path to the destination folder.

    Returns:
    - None
    """
    seg_folder = os.path.join(target_folder, 'seg')
    img_folder = os.path.join(target_folder, 'img')
    os.makedirs(seg_folder, exist_ok=True)
    os.makedirs(img_folder, exist_ok=True)

    for filename in os.listdir(source_folder):
        file_path = os.path.join(source_folder, filename)
        if os.path.isdir(file_path):
            continue  # Skip directories
        new_filename = filename.replace(" ", "")  # Remove spaces in filenames
        if 'seg' in new_filename.lower():  # Check if it's a segmentation file
            shutil.move(file_path, os.path.join(seg_folder, new_filename))
        else:  # Otherwise, treat as an image file
            shutil.move(file_path, os.path.join(img_folder, new_filename))

    messagebox.showinfo("Done", f"Files organized into {target_folder}")


def rename_files(directory):
    """
    Rename files to a standard format with ID and plane information.

    Args:
    - directory (str): Path to the directory containing files to rename.

    Returns:
    - None
    """
    for filename in os.listdir(directory):
        # 修改正则表达式以接受首字母为 'A' 或 'I'
        if match := re.match(r'([AI]\d+)([^\.]+)\.nii', filename, re.IGNORECASE):
            id_part = match.group(1).upper()  # Extract ID (e.g., A101, I101)
            plane_part = match.group(2)  # Extract plane information
            new_plane_part = 'unknown'
            if 'contrast' in plane_part.lower():
                new_plane_part = 'contrast'
            elif 'axial' in plane_part.lower():
                new_plane_part = 'axial'
            elif 'coronal' in plane_part.lower() or 'cor' in plane_part.lower():
                new_plane_part = 'coronal'
            # Create a new filename
            new_filename = f"{id_part} {new_plane_part}.nii.gz"
            old_file_path = os.path.join(directory, filename)
            new_file_path = os.path.join(directory, new_filename)

            # 检查目标文件是否已存在
            if os.path.exists(new_file_path):
                print(f"File {new_file_path} already exists. Skipping.")
                continue

            # 执行重命名
            os.rename(old_file_path, new_file_path)
            print(f"Renamed {old_file_path} to {new_file_path}")

    messagebox.showinfo("Done", "Files renamed successfully")



def tidy_plane(source_directory, target_directory, type='img'):
    """
    Classify files into subfolders based on plane type (axial, coronal, contrast).

    Args:
    - source_directory (str): Path to the folder containing files to classify.
    - target_directory (str): Path to the folder where files will be classified.
    - type (str): Subfolder type ('img' or 'seg').

    Returns:
    - None
    """
    for filename in os.listdir(source_directory):
        sub_target_folder = target_directory
        if 'axial' in filename.lower():
            sub_target_folder = os.path.join(sub_target_folder, 'axial', type)
        elif 'coronal' in filename.lower() or 'cor' in filename.lower():
            sub_target_folder = os.path.join(sub_target_folder, 'coronal', type)
        elif 'contrast' in filename.lower():
            sub_target_folder = os.path.join(sub_target_folder, 'contrast', type)
        else:
            continue  # Skip files without plane information
        os.makedirs(sub_target_folder, exist_ok=True)
        shutil.copy2(os.path.join(source_directory, filename), os.path.join(sub_target_folder, filename))

    messagebox.showinfo("Done", f"Files classified by plane into {target_directory}")


class GenerateCentrelinePoints:
    """
    Generate centreline points from segmentation masks and save them as text files.
    """
    def __init__(self, seg_folder, output_folder):
        self.seg_folder = seg_folder
        self.output_folder = output_folder

    def generate_centreline(self):
        """
        Generate and save centreline points for each segmentation file.
        """
        os.makedirs(self.output_folder, exist_ok=True)
        for filename in os.listdir(self.seg_folder):
            seg_path = os.path.join(self.seg_folder, filename)
            seg = sitk.ReadImage(seg_path)
            seg_arr = sitk.GetArrayFromImage(seg)
            centreline_points = self._generate_centreline_points(seg_arr)

            # Determine output filename
            base_name = filename.split('.')[0]  # Get the base name without extensions
            output_path = os.path.join(self.output_folder, base_name + '.txt')

            # Save centreline points to a .txt file
            with open(output_path, 'w') as file:
                for point in centreline_points:
                    file.write(f'{point[0]} {point[1]} {point[2]}\n')
        print(f"Centreline points saved in {self.output_folder}")

    def _generate_centreline_points(self, seg_arr):
        """
        Extract medial axis and return centreline points for each connected component.
        """
        centreline_points = []
        for slice_idx in tqdm(range(seg_arr.shape[0])):
            slice = seg_arr[slice_idx]
            if np.any(slice):  # If the slice is not empty
                labeled_slice, num_features = label(slice, return_num=True, connectivity=2)
                for i in range(1, num_features + 1):  # Process each connected component
                    component = labeled_slice == i
                    skeleton, distance = medial_axis(component, return_distance=True)
                    max_dist = distance.max()
                    central_points = np.argwhere(distance == max_dist)
                    if central_points.size > 0:
                        idx = len(central_points) // 2
                        central_point = central_points[idx]
                        centreline_points.append((slice_idx, int(central_point[0]), int(central_point[1])))
        return centreline_points


# ========================== Tkinter GUI ==========================

def select_folder(label):
    """
    Open a file dialog to select a folder and display the path on a label.
    """
    folder = filedialog.askdirectory()
    if folder:
        label.config(text=folder)


def run_overwrite_labels():
    src = src_folder_label.cget("text")
    tgt = tgt_folder_label.cget("text")
    overwrite_labels(src, tgt, OnlyTI=True)


def run_tidy_folder():
    src = src_folder_label.cget("text")
    tgt = tgt_folder_label.cget("text")
    tidy_folder(src, tgt)


def run_rename_files():
    directory = src_folder_label.cget("text")
    rename_files(directory)


def run_tidy_plane():
    src = src_folder_label.cget("text")
    tgt = tgt_folder_label.cget("text")
    tidy_plane(src, tgt, type='img')


def run_generate_centreline():
    src = src_folder_label.cget("text")
    tgt = tgt_folder_label.cget("text")
    generator = GenerateCentrelinePoints(src, tgt)
    generator.generate_centreline()


# ========================== Main GUI Layout ==========================

root = Tk()
root.title("Medical Image Processing Tool")

# Input Source Folder
Label(root, text="Source Folder:").grid(row=0, column=0, padx=10, pady=10)
src_folder_label = Label(root, text="", relief="solid", width=40)
src_folder_label.grid(row=0, column=1, padx=10, pady=10)
Button(root, text="Select", command=lambda: select_folder(src_folder_label)).grid(row=0, column=2, padx=10, pady=10)

# Input Target Folder
Label(root, text="Target Folder:").grid(row=1, column=0, padx=10, pady=10)
tgt_folder_label = Label(root, text="", relief="solid", width=40)
tgt_folder_label.grid(row=1, column=1, padx=10, pady=10)
Button(root, text="Select", command=lambda: select_folder(tgt_folder_label)).grid(row=1, column=2, padx=10, pady=10)

# Function Buttons
Button(root, text="Overwrite Labels", command=run_overwrite_labels).grid(row=2, column=0, padx=10, pady=10)
Button(root, text="Tidy Folder", command=run_tidy_folder).grid(row=2, column=1, padx=10, pady=10)
Button(root, text="Rename Files", command=run_rename_files).grid(row=2, column=2, padx=10, pady=10)
Button(root, text="Classify by Plane", command=run_tidy_plane).grid(row=3, column=0, padx=10, pady=10)
Button(root, text="Generate Centreline", command=run_generate_centreline).grid(row=3, column=1, padx=10, pady=10)

root.mainloop()
