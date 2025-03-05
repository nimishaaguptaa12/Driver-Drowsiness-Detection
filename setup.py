import os

folders = [
    "driver_drowsy",
    "driver_drowsy/train",
]

files = {
    "driver_drowsy/main.py": "",
    "driver_drowsy/model.py": "",
    "driver_drowsy/preprocess.py": "",
    "driver_drowsy/predict.py": "",
    "driver_drowsy/.gitignore": "drowsiness_model.h5\n__pycache__/",
}

for folder in folders:
    os.makedirs(folder, exist_ok=True)

for file_path, content in files.items():
    with open(file_path, "w") as f:
        f.write(content)

print("Project structure created successfully!")
