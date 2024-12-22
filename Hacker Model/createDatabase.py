import json
import subprocess
import os
import shutil

def resetDatabases():
  # Delete and recreate the databases.json file
  if os.path.exists("databases.json"):
    os.remove("databases.json")
    print("databases.json deleted successfully.")

  with open("databases.json", "w") as json_file:
    json.dump({"databases": []}, json_file)

  # Delete all_players_data.db if it exists
  if os.path.exists("all_players_data.db"):
    os.remove("all_players_data.db")
    print("all_players_data.db deleted successfully.")

  # Get the absolute path of the current Python file
  current_file_path = os.path.abspath(__file__)

  # Get the directory containing the Python file
  current_folder_path = os.path.dirname(current_file_path)

  # Get the path of the databases folder
  databases_folder_path = os.path.join(current_folder_path, "databases")

  # Delete the databases folder if it exists
  if os.path.exists(databases_folder_path):
    shutil.rmtree(databases_folder_path)
    print(f"Folder '{databases_folder_path}' deleted successfully.")

def createDatabaseFolder(data: list):
  # Create the databases folder
  for index, (steamID, isHacker) in enumerate(data):
    print(f"Processing Steam ID {steamID}, ({index + 1}/{len(data)}), isHacker: {isHacker}")

    # Depending on the platform, run the appropriate Python command
    if os.name == 'nt':
      subprocess.run(["python", "parseGameData.py", str(steamID), str(1), str(isHacker)])
    else:
      subprocess.run(["python3", "parseGameData.py", str(steamID), str(1), str(isHacker)])

# Reset the databases
resetDatabases()

with open("input_file.json", 'r') as json_file:
  data = json.load(json_file)

# Artificially filter the amount of data we want the model to be trained on
numberOfDataPoints = len(data)
first50DataPoints = data[:numberOfDataPoints]
second50DataPoints = data[numberOfDataPoints:]
third50DataPoints = data[numberOfDataPoints * 2:]
fourth50DataPoints = data[numberOfDataPoints * 3:]
last50DataPoints = data[numberOfDataPoints * 4:]

# Create the databases folder
createDatabaseFolder(first50DataPoints)
createDatabaseFolder(second50DataPoints)

# createDatabaseFolder(third50DataPoints)
# createDatabaseFolder(fourth50DataPoints)
# createDatabaseFolder(last50DataPoints)

# Merge the SQLite databases
print(f"Merging databases")
if os.name == 'nt':
  subprocess.run(["python", "mergeDatabase.py"])
else:
  subprocess.run(["python3", "mergeDatabase.py"])