import json
import sqlite3
import os

json_file_path = "databases.json"

with open(json_file_path, "r") as json_file:
  file_paths_data = json.load(json_file)
    
merged_db_conn = sqlite3.connect("all_players_data.db")
merged_db_cursor = merged_db_conn.cursor()

merged_db_cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = merged_db_cursor.fetchall()
for table in tables:
  merged_db_cursor.execute(f"DROP TABLE {table[0]}")

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

merged_db_cursor.execute(createTableQuery)
merged_db_conn.commit()

for dbFilePath in file_paths_data["databases"]:
  if os.path.exists(dbFilePath):
    db_conn = sqlite3.connect(dbFilePath)
    db_cursor = db_conn.cursor()
    
    db_cursor.execute("SELECT * FROM all_player_stats")
    data_to_merge = db_cursor.fetchall()
    
    insertQuery: str = """INSERT INTO all_player_stats VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)"""
    merged_db_cursor.executemany(insertQuery, data_to_merge)
    merged_db_conn.commit()
    db_conn.close()

merged_db_conn.close()