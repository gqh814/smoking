from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd


def scrape_mediestream(start_year, end_year):
    # Setup Chrome WebDriver
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # Run in headless mode (no GUI)
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920x1080")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    results = []
    base_url = "https://www2.statsbiblioteket.dk/mediestream/avis/search/rygning/date/{year}-01-01%2C{year}-12-31"

    for year in range(start_year, end_year + 1):
        
        url = base_url.format(year=year)
        driver.get(url)

        try:
            # Wait until at least one "searchInfoText" element is present
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, '//div[@id="searchinfo"]//div[contains(@class, "searchInfoText")]'))
            )
            
            # Select the last "searchInfoText" element
            hits_text_element = driver.find_element(By.XPATH, '(//div[@id="searchinfo"]//div[contains(@class, "searchInfoText")])[last()]')
            hits_text = hits_text_element.text
            print(year, hits_text)
            # Extract the number of hits
            hits_number = hits_text.split()[-2].replace(".", "")  # Extract number and remove dots
        except Exception:
            hits_number = "0"  # Default to 0 if no hits are found

        results.append({"year": year, "hits": int(hits_number)})

    driver.quit()

    return pd.DataFrame(results)


# Example usage:
if __name__ == "__main__":
    start_year = 1850
    end_year = 2010
    df = scrape_mediestream(start_year, end_year)
    df.to_csv(f"./data/scrape_mediestream_{start_year}_{end_year}.csv", index=False)
    print(df)
