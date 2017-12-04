# OCCS-board-game-project
Noel and Orion's final project for Machine Learning

grep "boardgamemechanic" ./test.txt | sed -E 's/.*<.*value=[\"](.*)[\"].*>/\1/' > output.txt

Things we want:

boardgamemechanic
boardgamecategory
boardgamefamily
averageweight

Command to extract the mechanics from a single game's XML file
Needs to be done for category, mechanics, and family (find out what XML tags are)
...for each game