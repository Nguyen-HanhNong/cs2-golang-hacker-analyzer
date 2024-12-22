from playwright.sync_api import sync_playwright
import random, time, sqlite3, sys, os, json

def scrapeSteamID(isTrainingData: bool, cheatingValue: int, steamID: str) -> list:
  with sync_playwright() as p:
    # Launch the browser (headless mode is disabled so that we can bypass Cloudflare's anti-bot protection)
    browser = p.chromium.launch(headless=False)

    # Create a browser context and a page
    context = browser.new_context(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36", viewport={"width": 480, "height": 360}, java_script_enabled=True)
    page = context.new_page()

    page.set_default_timeout(5000)  # For actions like click, wait_for_selector, etc.
    page.set_default_navigation_timeout(5000)  # For navigation actions like goto()
    page.wait_for_timeout(500) # To bypass errors in rate limiting

    # Navigate to a URL
    page.goto(f"https://csstats.gg/player/{steamID}", referer="https://google.com/")
    
    # Click on matches tab
    page.click("#matches-nav")

    # Check and accept cookie banner if present
    try:
      page.get_by_test_id("uc-accept-all-button").click(timeout=5000)
      print("Cookies accepted.")
    except Exception:
      print("No cookie banner found.")

    print("Scraping data...")

    # Get all <tr> locators within the <tbody>
    tr_locators = page.locator(f"tr[onclick*='window.location']").all()

    # Extract href from <a> inside each <tr> and store them in a list
    href_links = []
    for i, tr_locator in enumerate(tr_locators):
      link_locator = tr_locator.locator("a.match-list-link")  # Replace with the specific selector for <a>
      href = link_locator.get_attribute("href")
      if href:  # Ensure href is not None
          href_links.append(href)

    players_data = []

    # We need to curate the search to only the 5 most recent games, but there is a chance that the player has not played 10 games yet
    # We can use the length of href_links to determine the number of games played
    # If the player has played more than 5 games, we only need the 5 most recent games
    if len(href_links) > 10:
      href_links = href_links[:10]

    # Visit each URL in href_links
    for url in href_links:
      page.goto(url)

      # Extract the game file name from the URL
      gameID = url.split("/")[-1]

      page.get_by_role("cell", name=" Utility").locator("span").click(timeout=1000)
      page.get_by_role("cell", name=" First Kill").locator("span").click(timeout=1000)
      page.get_by_role("cell", name=" Trades").locator("span").click(timeout=1000)
      page.get_by_role("cell", name=" 1vX").locator("span").click(timeout=1000)
      page.get_by_role("cell", name=" Multikills").locator("span").click(timeout=1000)

      # Locate the table
      table_locator = page.locator("table#match-scoreboard")

      second_tbody = table_locator.locator("tbody").nth(0)  # Index starts at 0, so 1 = second <tbody>
      fourth_tbody = table_locator.locator("tbody").nth(2)  # Index 3 = fourth <tbody>

      # Search for the <td> tag with the matching SteamID in the first <tbody>
      check_for_player_in_tbody_2 = second_tbody.locator(f"a[href*='{steamID}']").locator("..").locator("..")
      check_for_player_in_tbody_4 = fourth_tbody.locator(f"a[href*='{steamID}']").locator("..").locator("..")

      player_tbody = None
      if check_for_player_in_tbody_2.count() > 0:
        player_tbody = check_for_player_in_tbody_2
      else:
        player_tbody = check_for_player_in_tbody_4

      # Extract player-specific information
      kills = player_tbody.locator("td:nth-child(6)").inner_text()  # Kills
      deaths = player_tbody.locator("td:nth-child(7)").inner_text()  # Deaths
      assists = player_tbody.locator("td:nth-child(8)").inner_text()  # Assists
      adr = player_tbody.locator("td:nth-child(11)").inner_text()  # ADR
      headshot_percentage = player_tbody.locator("td:nth-child(12)").inner_text().replace("%", "")  # Headshot percentage
      
      rating = player_tbody.locator("td:nth-child(14)").inner_text()  # Rating
      number_of_5k = player_tbody.locator("td:nth-child(41)").inner_text()
      number_of_4k = player_tbody.locator("td:nth-child(42)").inner_text()
      number_of_3k = player_tbody.locator("td:nth-child(43)").inner_text()
      number_of_2k = player_tbody.locator("td:nth-child(44)").inner_text()
      number_of_1k = player_tbody.locator("td:nth-child(45)").inner_text()

      # We check if the player is the player is a known cheater or if we don't know if the player is cheating (this handles the training data)
      is_cheating = cheatingValue if isTrainingData == True else -1

      # Store the player-specific information in a dictionary
      playerData = {
        "gameId": gameID,
        "steamId": steamID,
        "isCheating": is_cheating,
        "kills": kills,
        "deaths": deaths,
        "assists": assists,
        "adr": adr,
        "headshot_percentage": headshot_percentage,
        "rating": rating,
        "number_of_5k": number_of_5k,
        "number_of_4k": number_of_4k,
        "number_of_3k": number_of_3k,
        "number_of_2k": number_of_2k,
        "number_of_1k": number_of_1k,
      }

      with open('variables.json', 'r') as json_file:
        variables = json.load(json_file)
          
      noMissingVariables = all(variable in playerData for variable in variables)
      
      if noMissingVariables:
        # Append the player-specific information to the players_data list
        game_data = tuple(playerData.values())
        players_data.append(game_data)

    # Close the scraping browser
    browser.close()

    # print(f"Scraped data for player: {steamID} \n, Data: {players_data}")
    return players_data

def createDirectory(steamID: str) -> str:
  # Create a new folder for the databases if it doesn't exist
  newFolderForDatabase = "databases"
  try:
    os.mkdir(newFolderForDatabase)
  except OSError:
    print("Directory Already Exists")

  # Create a new directory inside the databases folder for the steamID
  databaseForSteamIDPath = f"{newFolderForDatabase}/{steamID}"
  
  try:  
    os.mkdir(databaseForSteamIDPath)
  except OSError:  
    print("Directory Already Exists")
    return False
  
  return databaseForSteamIDPath

def main(steamID: str, isTrainingData: int, isHacker: int = 0):
  gameData: list = scrapeSteamID(True if isTrainingData == 1 else False, isHacker, steamID)
  
  if gameData != None and len(gameData) > 0:
    directory: str = createDirectory(steamID)

    if directory:
      databasePath: str = os.path.join(directory, steamID+"_info.db")
    
      with open("databases.json", "r") as json_file:
        db_data = json.load(json_file)
        
      db_data["databases"].append(databasePath)
        
      with open("databases.json", "w") as json_file:
        json.dump(db_data, json_file, indent=4)
        
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

if __name__ == "__main__":
  if len(sys.argv) < 3:
    print("Usage: python parseGameData.py <steamID> <isTrainingData> <isHacker - optional>")
    sys.exit(1)

  steamID: str = sys.argv[1]
  isTrainingData: int = int(sys.argv[2])

  if len(sys.argv) > 3:
    isHacker: int = int(sys.argv[3])
    main(steamID, isTrainingData, isHacker)
  else:
    main(steamID, isTrainingData)
    
# Example Command 1 for parsing an account that is training data and is a hacker: python parseGameData 76561198438276637 1 1
# Example Command 2 for parsing an account that is training data and isn't a hacker: python parseGameData 76561198438276637 1 0
# Example Command 3 for parsing an account that isn't training data: python parseGameData 76561198438276637 0