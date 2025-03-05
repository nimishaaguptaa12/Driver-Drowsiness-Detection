import os

# Define the base directory
base_dir = "mobileNet_Model"

# Define subdirectories
sub_dirs = [
    "train",
    "test",
    "models"
]

# Define Python script files
script_files = [
    "preprocess.py",
    "mobilenet_model.py",
    "train.py",
    "predict.py"
]

# Create base directory if it doesn't exist
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

# Create subdirectories
for sub_dir in sub_dirs:
    os.makedirs(os.path.join(base_dir, sub_dir), exist_ok=True)

# Create empty script files
for script in script_files:
    script_path = os.path.join(base_dir, script)
    if not os.path.exists(script_path):
        with open(script_path, "w") as f:
            f.write("# " + script + " - Auto-generated file\n")

print(f"Project structure created under '{base_dir}'")
