import requests
from urllib.parse import urlsplit, parse_qsl


api_key = "67cbdf6068110b0096bf71bf387b93d6a40b88e0b92ba2cf84e0ed47aef5da3b"
# place_id = "ChIJpZ4cTQ1544kReshVAg1dLg4"
query = "Tatte Bakery & Cafe Boston"
data_id = "0x89e37a74dd161527:0xb69e3faf74af50b3"



base_url = "https://serpapi.com/search.json"

params = {
    "engine": "google_maps_reviews",
    "data_id": data_id,
    "api_key": api_key,
    "hl": "en",
    "sorta_by": "qualityScore"
}


all_reviews = []
while True:
    resp = requests.get(base_url, params=params)
    data = resp.json()
    reviews = data.get("reviews", [])
    all_reviews.extend(reviews)

    # Check if there is a next page
    pagination = data.get("serpapi_pagination", {})
    next_url = pagination.get("next")
    next_token = pagination.get("next_page_token")

    if next_url and next_token:
        # Parse out the params from `next` (it has the right data_id, token, etc)
        new_qs = dict(parse_qsl(urlsplit(next_url).query))
        params.update(new_qs)
    else:
        break

print("Number of reviews fetched:", len(all_reviews))
for review in all_reviews:
    print("User:", review.get("user", {}).get("name"))
    print("Rating:", review.get("rating"))
    print("Date:", review.get("iso_date"))
    print("Review:", review.get("snippet") or review.get("text", ""))
    print("---")
