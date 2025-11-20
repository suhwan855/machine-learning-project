import json
import requests

geo_url = "https://raw.githubusercontent.com/southkorea/southkorea-maps/master/kostat/2013/json/skorea_provinces_geo.json"

response = requests.get(geo_url)
geo_data = response.json()

for f in geo_data["features"]:
    print(f["properties"])
