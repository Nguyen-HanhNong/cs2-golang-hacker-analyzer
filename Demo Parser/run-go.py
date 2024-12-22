import subprocess
import os
from pathlib import Path

# Step 1: Empty all the files in the parsed-demo folder 

# Path to the folder containing the parsed JSON files
current_directory = os.path.dirname(os.path.abspath(__file__))
parsed_demo_folder = Path(os.path.join(current_directory, "parsed-demo"))

# Loop through all the files in the folder and delete them
for filename in os.listdir(parsed_demo_folder):
    file_path = os.path.join(parsed_demo_folder, filename)

    try:
        # Check if it's a file before deleting
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Deleted file: {file_path}")
    except Exception as e:
        print(f"Error deleting file {file_path}: {e}")

# Check if the folder is empty
if not os.listdir(parsed_demo_folder):
    print("All files deleted successfully!")
else:
    print("Error deleting files!")
    exit(-1)


# Step 2: Loop through all the demos in the "demos" folder so that we can get the parsed JSON data for each demo
folder_path = Path(os.path.join(current_directory, "demos"))
all_demo_file_names = [file.name for file in Path(folder_path).glob("*.dem")]

for demo_name in all_demo_file_names:
    print(demo_name)

    # Path to your Go script or compiled binary
    go_script_path = "main.go"

    # Command-line arguments to pass
    flags = ["demos", demo_name]

    # Combine the script and arguments
    command = ["go", "run", go_script_path] + flags

    try:
        # Run the Go script and capture output
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print("Output:\n", result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error:\n", e.stderr)