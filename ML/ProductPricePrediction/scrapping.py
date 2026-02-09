import requests
from bs4 import BeautifulSoup
import pandas as pd

headers = {
    "User-Agent": "Mozilla/5.0"
}

url = "https://www.flipkart.com/search?q=mobiles&as=on&as-show=on&otracker=AS_Query_HistoryAutoSuggest_1_6_na_na_na&otracker1=AS_Query_HistoryAutoSuggest_1_6_na_na_na&as-pos=1&as-type=HISTORY&suggestionId=mobiles&requestId=a4b17236-22a6-4fa3-800b-37acc52f0ff0&as-searchtext=mobile&page=1"
response = requests.get(url, headers=headers, verify=False)
soup = BeautifulSoup(response.text, "html.parser")

products = []

for item in soup.select("div._1AtVbE"):
    name = item.select_one("div._4rR01T")
    price = item.select_one("div._30jeq3")
    rating = item.select_one("div._3LWZlK")

    if name and price:
        products.append({
            "name": name.text,
            "price": price.text.replace("₹", "").replace(",", ""),
            "rating": rating.text if rating else None
        })

df = pd.DataFrame(products)
df["price"] = df["price"].astype(int)

df.to_csv("flipkart_mobiles.csv", index=False)
print(df.head())
