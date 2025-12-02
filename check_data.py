import os
data_dir = "chest_xray"

for split in ["train", "val", "test"]:
    split_dir = os.path.join(data_dir, split)
    for category in ["PNEUMONIA", "NORMAL"]:
        category_dir = os.path.join(split_dir, category)
        num_files = len(os.listdir(category_dir))
        print(f"Number of images in {split}/{category}: {num_files}")