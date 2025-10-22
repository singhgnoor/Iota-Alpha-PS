import os
import shutil
import random
import glob

# -----------------------------------------------------------------------------
# --- CONFIGURATION: YOU MUST EDIT THESE 4 PATHS ---
# -----------------------------------------------------------------------------

# 1. Path to the folder where all your downloaded IMAGES are
#    (e.g., 'C:/Users/YourName/Downloads/KaggleDataset/images')
SOURCE_IMAGES_DIR = 'D:\Programming\Playground\datasets\FLY_MOS_Dataset'

# 2. Path to the folder where all your downloaded LABELS (.txt files) are
#    (e.g., 'C:/Users/YourName/Downloads/KaggleDataset/labels')
SOURCE_LABELS_DIR = 'D:\Programming\Playground\datasets\FLY_MOS_Dataset'

# 3. Path to the NEW dataset folder you want to create
#    (This script will create this folder and all subfolders)
DEST_DATASET_DIR = 'D:\Programming\Playground\datasets\Fly_Mos_Formatted'

# 4. Split ratio for training (e.g., 0.8 means 80% train, 20% validation)
TRAIN_SPLIT = 0.8


# -----------------------------------------------------------------------------
# --- END OF CONFIGURATION ---
# -----------------------------------------------------------------------------


def split_dataset(source_img_dir, source_label_dir, dest_dir, train_split):
    """
    Randomly splits a YOLO dataset into training and validation sets.
    """
    print(f"Starting dataset split...")
    print(f"Source Images: {source_img_dir}")
    print(f"Source Labels: {source_label_dir}")
    print(f"Destination: {dest_dir}")

    # --- 1. Create destination directories ---
    print("\nCreating destination directories...")
    train_img_path = os.path.join(dest_dir, 'images', 'train')
    val_img_path = os.path.join(dest_dir, 'images', 'val')
    train_label_path = os.path.join(dest_dir, 'labels', 'train')
    val_label_path = os.path.join(dest_dir, 'labels', 'val')

    os.makedirs(train_img_path, exist_ok=True)
    os.makedirs(val_img_path, exist_ok=True)
    os.makedirs(train_label_path, exist_ok=True)
    os.makedirs(val_label_path, exist_ok=True)
    print("Directories created successfully.")

    # --- 2. Find all image files and create (image, label) pairs ---
    print("\nFinding image and label pairs...")

    # Use glob to find all common image types
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(glob.glob(os.path.join(source_img_dir, ext)))

    print(f"Found {len(image_files)} total images.")
    if not image_files:
        print(f"ERROR: No images found in '{source_img_dir}'. Please check your path.")
        return

    file_pairs = []
    missing_labels = 0
    for img_path in image_files:
        # Get the base filename without the extension (e.g., 'img_001')
        base_filename = os.path.splitext(os.path.basename(img_path))[0]

        # Create the corresponding label file path
        label_path = os.path.join(source_label_dir, base_filename + '.txt')

        # Check if the label file exists
        if os.path.exists(label_path):
            file_pairs.append((img_path, label_path))
        else:
            print(f"Warning: Missing label for {img_path}. Skipping this file.")
            missing_labels += 1

    print(f"Found {len(file_pairs)} valid image/label pairs.")
    if missing_labels > 0:
        print(f"Skipped {missing_labels} images due to missing labels.")

    # --- 3. Shuffle the list randomly ---
    random.seed(42)  # Use a fixed seed for reproducible splits
    random.shuffle(file_pairs)
    print("Shuffled file pairs.")

    # --- 4. Split the list into train and val sets ---
    split_index = int(len(file_pairs) * train_split)
    train_files = file_pairs[:split_index]
    val_files = file_pairs[split_index:]

    print(f"Training set size: {len(train_files)} files")
    print(f"Validation set size: {len(val_files)} files")

    # --- 5. Copy files to their new homes ---
    def copy_files(file_list, img_dest, label_dest):
        count = 0
        for img_src, label_src in file_list:
            try:
                shutil.copy(img_src, img_dest)
                shutil.copy(label_src, label_dest)
                count += 1
            except Exception as e:
                print(f"ERROR copying {img_src}: {e}")
        return count

    print("\nCopying training files...")
    train_count = copy_files(train_files, train_img_path, train_label_path)
    print(f"Copied {train_count} training files.")

    print("\nCopying validation files...")
    val_count = copy_files(val_files, val_img_path, val_label_path)
    print(f"Copied {val_count} validation files.")

    print("\n--- SCRIPT COMPLETE ---")
    print(f"New dataset created at: {os.path.abspath(dest_dir)}")


# --- This line runs the function when you execute the script ---
if __name__ == "__main__":
    split_dataset(SOURCE_IMAGES_DIR, SOURCE_LABELS_DIR, DEST_DATASET_DIR, TRAIN_SPLIT)