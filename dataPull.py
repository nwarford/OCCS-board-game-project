import requests

URL = "https://www.boardgamegeek.com/xmlapi/boardgame/174430"

response = requests.get(URL)

testText = "test.txt"

with open(testText, 'wb+') as file:
    file.write(response.content)
