import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com/"
}


SEARCH_TERM = "headphones"
BASE_URL = f"https://www.amazon.in/s?k={SEARCH_TERM}&page={{}}"

all_products = []

for page in range(1, 4):  
    print(f"\n Scraping page {page} for {SEARCH_TERM}...")
    url = BASE_URL.format(page)
    response = requests.get(url, headers=HEADERS)
    soup = BeautifulSoup(response.text, "html.parser")

    products = soup.find_all("div", {"data-component-type": "s-search-result"})

    for product in products:
        # Product Name (try multiple selectors)
        title_tag = (
            product.select_one("span.a-size-medium.a-color-base.a-text-normal")
            or product.find("h2")
        )
        title = title_tag.get_text(strip=True) if title_tag else "N/A"

        # Price
        price_tag = product.find("span", {"class": "a-price-whole"})
        price = price_tag.get_text(strip=True).replace(",", "") if price_tag else "N/A"

        # MRP (original price)
        mrp_tag = product.find("span", {"class": "a-text-price"})
        mrp = mrp_tag.get_text(strip=True).replace(",", "") if mrp_tag else "N/A"

        # Discount
        discount = "N/A"
        discount_tag = product.select_one("span.a-letter-space + span")
        if discount_tag:
            discount = discount_tag.get_text(strip=True)

        # Rating
        rating_tag = product.find("span", {"class": "a-icon-alt"})
        rating = rating_tag.get_text(strip=True) if rating_tag else "N/A"

        # Reviews (try different selectors)
        review_tag = (
            product.find("span", {"class": "a-size-base"})
            or product.find("span", {"class": "a-size-base s-underline-text"})
        )
        reviews = review_tag.get_text(strip=True) if review_tag else "N/A"

        # Append data
        all_products.append({
            "Product Name": title,
            "Price (₹)": price,
            "MRP (₹)": mrp,
            "Discount": discount,
            "Rating": rating,
            "Reviews": reviews
        })

    time.sleep(2)

# Save to CSV
df = pd.DataFrame(all_products)
df.to_csv(f"amazon_{SEARCH_TERM}.csv", index=False, encoding="utf-8")
print(f"\n✅ Data saved to amazon_{SEARCH_TERM}.csv")
