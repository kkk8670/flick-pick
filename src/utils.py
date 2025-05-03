import os, sys
import shutil


# def init_env():
#   from dotenv import load_dotenv
#   load_dotenv()
#   root_dir = os.getenv("ROOT_DIR")
#   if root_dir and root_dir not in sys.path:
#       sys.path.insert(0, root_dir)
#     return root_dir


def save_to_csv(folder_path):
    path, name = os.path.dirname(folder_path), os.path.basename(folder_path)
    name = name if name.endswith(".csv") else f"{name}.csv"
    target_file = os.path.join(path, name)

    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            source_file = os.path.join(folder_path, filename)
            shutil.move(source_file, target_file)
    os.path.isdir(folder_path) and shutil.rmtree(folder_path)

    print(f"saved file: {target_file}")