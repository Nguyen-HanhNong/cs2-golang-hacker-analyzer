import sys, os, stat, shutil, sqlite3, logging, warnings
import tensorflow as tf
import shutil
import numpy as np
from parseGameData import scrapeSteamID, createDirectory

# Disable TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel(logging.ERROR)

# Disable scikit-learn warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

def grantPermissions(directory):
  for root, dirs, files in os.walk(directory, topdown=False):
    # Change permissions for files
    for name in files:
      file_path = os.path.join(root, name)
      try:
        os.chmod(file_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)  # Read, Write, Execute for all
        print(f"Permissions granted for file: {file_path}")
      except Exception as e:
        print(f"Failed to change permissions for file: {file_path}. Error: {e}")

    # Change permissions for directories
    for name in dirs:
      dir_path = os.path.join(root, name)
      try:
        os.chmod(dir_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)  # Read, Write, Execute for all
        print(f"Permissions granted for directory: {dir_path}")
      except Exception as e:
        print(f"Failed to change permissions for directory: {dir_path}. Error: {e}")

  # Finally, change permissions for the root directory
  try:
    os.chmod(directory, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
    print(f"Permissions granted for root directory: {directory}")
  except Exception as e:
    print(f"Failed to change permissions for root directory: {directory}. Error: {e}")



def main(steamID: str):
  # Delete the databases folder if it exists
  currentFilePath = os.path.abspath(__file__)
  currentFolderPath = os.path.dirname(currentFilePath)
  databasesFolderPath = os.path.join(currentFolderPath, "databases")

  if os.path.exists(databasesFolderPath):
    grantPermissions(databasesFolderPath)
    shutil.rmtree(databasesFolderPath)
    print(f"Folder '{databasesFolderPath}' deleted successfully.")

  directory = createDirectory(steamID)
  
  if directory:
    gameData: list = scrapeSteamID(False, -1, steamID)

    if gameData != None and gameData != [] and len(gameData) > 0:
      databasePath: str = os.path.join(directory, steamID+"_info.db")
      conn = sqlite3.connect(databasePath)
      cursor = conn.cursor()
      
      createTableQuery: str = """
        CREATE TABLE IF NOT EXISTS all_player_stats (
          gameId STRING,
          steamId STRING,
          isCheating INTEGER,
          kills INTEGER,
          deaths INTEGER,
          assists INTEGER,
          adr REAL,
          headshot_percentage REAL,
          rating REAL,
          number_of_5k INTEGER,
          number_of_4k INTEGER,
          number_of_3k INTEGER,
          number_of_2k INTEGER,
          number_of_1k INTEGER
        );
      """
      cursor.execute(createTableQuery)
      
      insertQuery: str = """INSERT INTO all_player_stats VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)"""
      cursor.executemany(insertQuery, gameData)
      conn.commit()
      conn.close()
        
      conn = sqlite3.connect(databasePath)
      cursor = conn.cursor()
      cursor.execute('SELECT * FROM all_player_stats')
      trainingData = cursor.fetchall()
      conn.close()
      
      X = np.array([row[3:] for row in trainingData])
      games = np.array([row[0] for row in trainingData])

      models_location = os.path.join(currentFolderPath, "models")
      models_file_name = f"nn_model_0.30.h5"

      # Run the model with the mean squared error loss
      loaded_model = tf.keras.models.load_model(os.path.join(models_location, models_file_name))
      loaded_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
      predictions = loaded_model.predict(X)

      # Print out the games where the player was likely cheating under the mean squared error model
      counter = 0
      print("\nThis player was likely cheating in the following games under the mean squared error model:\n------------------------------------------------")
      for i in range(len(predictions)):
        if predictions[i] > 0.3:
          print(f"https://csstats.gg/match/{games[i]}, prediction accuracy: {predictions[i]}")
          counter += 1
      
      print(f"Overall percentage chance of being a hacker/cheating: {counter/len(games)*100}%")
      print()
      print()
      print()

      # Delete the databases folder if it exists
      if os.path.exists(databasesFolderPath):
        grantPermissions(databasesFolderPath)
        shutil.rmtree(databasesFolderPath)
        print(f"Folder '{databasesFolderPath}' deleted successfully.")
          
if __name__ == "__main__":
  if len(sys.argv) != 2:
    print("Usage: python main.py <steamID>")
    print("Example: python main.py 76561198996682727")
    sys.exit(1)

  steamID: str = sys.argv[1]
  main(steamID)