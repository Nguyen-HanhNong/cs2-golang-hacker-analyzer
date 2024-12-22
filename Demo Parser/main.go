package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"time"

	demoinfocs "github.com/markus-wa/demoinfocs-golang/v4/pkg/demoinfocs"
	common "github.com/markus-wa/demoinfocs-golang/v4/pkg/demoinfocs/common"
	events "github.com/markus-wa/demoinfocs-golang/v4/pkg/demoinfocs/events"
)

func main() {
	args := os.Args

	if len(args) < 3 {
		fmt.Println("No arguments provided.")
		return
	}

	demoHashmap := make(map[string]interface{})
	demoHashmap["data"] = make([]string, 0)

	folder := args[1]
	fileName := args[2]
	relativePath := filepath.Join(folder, fileName)

	// Open the demo file
	f, err := os.Open(relativePath)
	checkError(err)
	defer f.Close()

	saveStartGameStatsToHashmap(demoHashmap, f)
	getNameOfAllPlayers(demoHashmap, f)
	// saveRoundStatsToHashmap(demoHashmap, f)
	// savePostGameStatsToHashmap(demoHashmap, f)

	// Create the JSON file to store the data
	currentTime := time.Now()
	timestampString := currentTime.Format("2006-01-02_15-04-05")
	saveDataToFile(demoHashmap, fmt.Sprintf("parsed_%s.json", timestampString), "parsed-demo")
}

func formatPlayer(p *common.Player) string {
	if p == nil {
		return "?"
	}

	switch p.Team {
	case common.TeamTerrorists:
		return "[T]" + p.Name
	case common.TeamCounterTerrorists:
		return "[CT]" + p.Name
	}

	return p.Name
}

func checkError(err error) {
	if err != nil {
		panic(err)
	}
}

func saveStartGameStatsToHashmap(demoHashmap map[string]interface{}, file *os.File) {

}

func getNameOfAllPlayers(demoHashmap map[string]interface{}, file *os.File) {
	// Create a parser for the demo file
	parser := demoinfocs.NewParser(file)

	// Ensure the parser is closed at the end
	defer parser.Close()

	// Maps to store unique players grouped by teams
	terrorists := make(map[uint64]string)
	counterTerrorists := make(map[uint64]string)

	// Parse the demo
	gameState := parser.GameState()

	for {
		// Advance the parser to the next event
		ok, err := parser.ParseNextFrame()
		if err != nil {
			log.Fatalf("Error parsing demo: %v", err)
		}
		if !ok {
			break // End of demo
		}

		// Continuously update players grouped by teams
		for _, player := range gameState.Participants().All() {
			if player == nil || !player.IsConnected {
				continue
			}

			// Check if the player is already part of one of the teams
			if _, inTerrorists := terrorists[player.SteamID64]; inTerrorists {
				continue
			}
			if _, inCounterTerrorists := counterTerrorists[player.SteamID64]; inCounterTerrorists {
				continue
			}

			// Group players by their teams
			switch player.Team {
				case common.TeamTerrorists:
					terrorists[player.SteamID64] = player.Name
				case common.TeamCounterTerrorists:
					counterTerrorists[player.SteamID64] = player.Name
			}
		}
	}

	// Get the two teams who played and save it to the hashmap
	teamT := parser.GameState().TeamTerrorists()
	teamCT := parser.GameState().TeamCounterTerrorists()

	teamTName := teamT.ClanName()
	teamCTName := teamCT.ClanName()

	demoHashmap[teamTName] = make([]string, 0)
	demoHashmap[teamCTName] = make([]string, 0)

	// Print grouped players from the last round
	fmt.Println("Terrorists (Last Round):")
	for steamID, name := range terrorists {
		fmt.Printf("SteamID: %d, Name: %s\n", steamID, name)
		demoHashmap[teamTName] = append(demoHashmap[teamTName].([]string), fmt.Sprintf("SteamID: %d, Name: %s\n", steamID, name))
	}

	fmt.Println("\nCounter-Terrorists (Last Round):")
	for steamID, name := range counterTerrorists {
		fmt.Printf("SteamID: %d, Name: %s\n", steamID, name)
		demoHashmap[teamCTName] = append(demoHashmap[teamCTName].([]string), fmt.Sprintf("SteamID: %d, Name: %s\n", steamID, name))
	}
}

func saveRoundStatsToHashmap(demoHashmap map[string]interface{}, file *os.File) {
	/* Get pointer to the parser */
	parser := demoinfocs.NewParser(file)
	defer parser.Close()

	// Register all of the handlers 
	parser.RegisterEventHandler(func(e events.Kill) {
		var hs string
		if e.IsHeadshot {
			hs = " (HS)"
		}
		var wallBang string
		if e.PenetratedObjects > 0 {
			wallBang = " (WB)"
		}
		fmt.Printf("%s <%v%s%s> %s\n", formatPlayer(e.Killer), e.Weapon, hs, wallBang, formatPlayer(e.Victim))
		demoHashmap["data"] = append(demoHashmap["data"].([]string), fmt.Sprintf("%s {%v%s%s} %s", formatPlayer(e.Killer), e.Weapon, hs, wallBang, formatPlayer(e.Victim)))
	})

	// Register handler on round end to figure out who won
	parser.RegisterEventHandler(func(e events.RoundEnd) {
		gs := parser.GameState()
		switch e.Winner {
		case common.TeamTerrorists:
			// Winner's score + 1 because it hasn't actually been updated yet
			fmt.Printf("Round finished: winnerSide=T  ; score=%d:%d\n", gs.TeamTerrorists().Score(), gs.TeamCounterTerrorists().Score())
			demoHashmap["data"] = append(demoHashmap["data"].([]string), fmt.Sprintf("Round finished: winnerSide=T  ; score=%d:%d", gs.TeamTerrorists().Score(), gs.TeamCounterTerrorists().Score()))
		case common.TeamCounterTerrorists:
			fmt.Printf("Round finished: winnerSide=CT ; score=%d:%d\n", gs.TeamCounterTerrorists().Score(), gs.TeamTerrorists().Score())
			demoHashmap["data"] = append(demoHashmap["data"].([]string), fmt.Sprintf("Round finished: winnerSide=CT ; score=%d:%d", gs.TeamCounterTerrorists().Score(), gs.TeamTerrorists().Score()))
		default:
			// Probably match medic or something similar
			fmt.Println("Round finished: No winner (tie)")
			demoHashmap["data"] = append(demoHashmap["data"].([]string), "Round finished: No winner (tie)")
		}
	})

	// Parse straight to the end
	err := parser.ParseToEnd()
	checkError(err)
}

func savePostGameStatsToHashmap(demoHashmap map[string]interface{}, file *os.File) {
	// Create parser for the demo file
	parser := demoinfocs.NewParser(file)
	defer parser.Close()

	// Parse straight to the end
	err := parser.ParseToEnd()
	checkError(err)

	// Get the two teams who played and save it to the hashmap
	teamT := parser.GameState().TeamTerrorists()
	teamCT := parser.GameState().TeamCounterTerrorists()

	teamTName := teamT.ClanName()
	teamCTName := teamCT.ClanName()

	fmt.Println("Team T:", teamTName)
	fmt.Println("Team CT:", teamCTName)

	demoHashmap["teams"] = make([]string, 0)
	demoHashmap["teams"] = append(demoHashmap["teams"].([]string), fmt.Sprintf("Team T: %s", teamTName))
	demoHashmap["teams"] = append(demoHashmap["teams"].([]string), fmt.Sprintf("Team CT: %s", teamCTName))

	// Get the players for each team and save it to the hashmap under the respective team
	teamTPlayers := teamT.Members()
	teamCTPlayers := teamCT.Members()

	demoHashmap["teamTPlayers"] = make([]string, 0)
	demoHashmap["teamCTPlayers"] = make([]string, 0)

	for _, player := range teamTPlayers {
		fmt.Println("Team T player:", player.Name)
		demoHashmap["teamTPlayers"] = append(demoHashmap["teamTPlayers"].([]string), player.Name)
	}

	for _, player := range teamCTPlayers {
		fmt.Println("Team CT player:", player.Name)
		demoHashmap["teamCTPlayers"] = append(demoHashmap["teamCTPlayers"].([]string), player.Name)
	}


	// Get each player's stats and save it to the hashmap under the respective team
	demoHashmap["teamTStats"] = make([]string, 0)
	demoHashmap["teamCTStats"] = make([]string, 0)

	for _, player := range teamTPlayers {
		fmt.Println("Team T player stats:", player.Name)
		fmt.Println("Kills:", player.Kills())
		fmt.Println("Deaths:", player.Deaths())
		fmt.Println("Assists:", player.Assists())

		demoHashmap["teamTStats"] = append(demoHashmap["teamTStats"].([]string), fmt.Sprintf("%s: %d kills, %d deaths, %d assists", player.Name, player.Kills(), player.Deaths(), player.Assists()))
	}

	for _, player := range teamCTPlayers {
		fmt.Println("Team CT player stats:", player.Name)
		fmt.Println("Kills:", player.Kills())
		fmt.Println("Deaths:", player.Deaths())
		fmt.Println("Assists:", player.Assists())

		demoHashmap["teamCTStats"] = append(demoHashmap["teamCTStats"].([]string), fmt.Sprintf("%s: %d kills, %d deaths, %d assists", player.Name, player.Kills(), player.Deaths(), player.Assists()))
	}

	// Get the final score of the game and save it to the hashmap
	finalScore := fmt.Sprintf("Final score: %d:%d", teamT.Score(), teamCT.Score())
	fmt.Println(finalScore)
	demoHashmap["finalScore"] = finalScore
}

func saveDataToFile(data map[string]interface{}, fileName string, directory string) {
	// Create the file path to the new JSON file
	filePath := filepath.Join(directory, fileName)

	// Create the directory if it doesn't exist
	err := os.MkdirAll(directory, os.ModePerm)
	if err != nil {
		fmt.Println("Error creating directory:", err)
		return
	}

	file, err := os.Create(filePath)
	if err != nil {
		fmt.Println("Error creating file:", err)
		return
	}
	defer file.Close()

	// Encode the data into the file
	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ") // Pretty print the JSON
	if err := encoder.Encode(data); err != nil {
		fmt.Println("Error writing to file:", err)
		return
	}

	fmt.Println("JSON file created successfully!")
}