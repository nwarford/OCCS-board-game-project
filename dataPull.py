import requests

URL = "https://www.boardgamegeek.com/xmlapi2/thing"

inputDict = {'id':161936, 'stats':1}

response = requests.get(URL,inputDict)

testText = "test.txt"

with open(testText, 'wb+') as file:
    file.write(response.content)
