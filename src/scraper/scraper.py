import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import logging
from pathlib import Path
from typing import Optional

#set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)

#header to not get banned by the SEC
SEC_HEADERS = {
    "User-Agent": "fumble fumble fumble@fumble.com",
    "Accept-Encoding": "gzip, deflate"
}

# wikipedia per-year Chapter 11 category page 
WIKI_CH11_URL = (
    "https://en.wikipedia.org/wiki/Category:"
    "Companies_that_filed_for_Chapter_11_bankruptcy_in_{year}"
)

"""
A bit of background:
SIC code ranges can be used for supply chain tier mappign
Tier 0: raw materials/mining/agriculture
Tier 1: Basic manufacturing/chemicals
Tier 2: Components/electronics
Tier 3: Assembly/finished goods
Tier 4: Retail/distribution/wholesale
"""

SIC_RANGES = [
    (range(100, 1000), 0),
    (range(2000, 2800), 1),
    (range(2800, 3600), 1),
    (range(3000, 3600), 3),
    (range(3600, 3800), 2),
    (range(3800, 4000), 2),
    (range(5000, 5200), 3),
    (range(5200, 5400), 4),
    (range(5400, 5800), 4),
    (range(5900, 6000), 4),
]

def sic_to_tier(sic: int) -> int:
    """map SIC to supply chain tier (the lower, the higher upstream)"""
    for sic_range, tier in SIC_RANGES:
        if sic in sic_range:
            return tier
    
    return 2 #default to middle if unknown

class DisruptionLabelScraper:
    """scrapes chapter 11 bankruptcy company names from wikipedia"""
    def __init__(self, output_dir: str = "src/data/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update(SEC_HEADERS)

    def scrape_wikipedia_bankruptcies(self, years: list[int] | None = None) -> pd.DataFrame:
        """
        Scrape Wikipedia's per-year Chapter 11 category pages.
        Each page has company names inside <div id="mw-pages">.

        Args:
            years: ears to scrape by default we only consider 2018 to 2025

        Returns:
            dataframe with columns: name, year, disrupted, source
        """

        if years is None:
            years = list(range(2018, 2026))

        all_records = []

        for year in years: 
            url = WIKI_CH11_URL.format(year=year)
            logger.info(f"Scraping Wikipedia chapter 11 for {year} : {url}")

            try:
                resp = self.session.get(url, timeout=15)
                resp.raise_for_status()

            except requests.HTTPError as e:
                logger.warning(f" HTTP {e.response.status_code} for {year}, skipping")
                continue

            except Exception as e:
                logger.warning(f" Failed {year}: {e}, skipping")
                continue

            soup = BeautifulSoup(resp.content, "html.parser")

            #all category members are inside this paragraph <div id="mw-pages">
            mw_pages = soup.find("div", {"id": "mw-pages"})
            if not mw_pages:
                logger.warning(f" No mw-pages div found for {year}")
                continue

            year_count = 0
            
            for link in mw_pages.find_all("a"):
                name = link.get_text(strip=True)
                if name:
                    all_records.append({
                        "name": name,
                        "year": year,
                        "disrupted": 1,
                        "source": "wikipedia_ch11",
                    })
                    year_count += 1
            
            logger.info(f" {year_count} companies found for {year}")

            time.sleep(1) #nice delay for wikipedia
        
        df = pd.DataFrame(all_records)
        out_path = self.output_dir / "bankruptcies.csv"
        df.to_csv(out_path, index=False)
        logger.info(f"Saved {len(df)} total records to {out_path}")
        return df
    
    def builld_disrupted_name_set(self, years: list[int] | None = None) -> set[str]:
        """return flat uppercased set of all bankrupt company names"""
        df = self.scrape_wikipedia_bankruptcies(years=years)
        if df.empty:
            return set()
        return set(df["name"].str.upper().str.strip().dropna()) #giga ugly but yeah sure why not
    
class SECEdgarScraper:
    """fetch company metadata from SEC EDGAR"""

    TICKERs_URL = "https://www.sec.gov/files/company_tickers.json"
    SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"

    def __init__(self, output_dir: str = "src/data/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update(SEC_HEADERS)

    def fetch_all_tickers(self, max_companies: int=2500) -> pd.DataFrame:
        
        logger.info("Fetching tickers from SEC EDGAR ...")  
        resp = self.session.get(self.TICKERs_URL, timeout=15)
        resp.raise_for_status()

        records = [
            {
                "cik": str(c["cik_str"]).zfill(10),
                "ticker": c["ticker"],
                "name": c["title"],
            }
            for i, c in list(resp.json().items())[:max_companies]
        ]

        return pd.DataFrame(records)
    
    def fetch_company_metadata(self, cik: str) -> dict | None:
        
        url = self.SUBMISSIONS_URL.format(cik=cik)
        
        try:
            resp = self.session.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            forms = data.get("filings", {}).get("recent", {}).get("form", [])
            sic = int(data.get("sic", 0) or 0)

            return {
                "cik":             cik,
                "name":            data.get("name", ""),
                "sic":             sic,
                "sic_description": data.get("sicDescription", ""),
                "state":           data.get("stateOfIncorporation", ""),
                "n_10k":           forms.count("10-K"),
                "n_8k":            forms.count("8-K"),
                "n_total_filings": len(forms),
                "tier":            sic_to_tier(sic),
            }
        
        except Exception as e:
            logger.debug(f"Failed CIK {cik}: {e}")
            return None
        
    def build_company_dataset(self, max_companies: int=2500) -> pd.DataFrame:
        tickers = self.fetch_all_tickers(max_companies)
        records = []
        for i, row in tickers.iterrows():
            meta = self.fetch_company_metadata(row["cik"])
            if meta:
                meta["ticker"] = row["ticker"]
                records.append(meta)
            time.sleep(0.12) #time limit required byt the SEC

            if (i + 1) % 50 == 0:
                logger.info(f" Processed {i+1}/{max_companies} ...")
        
        df = pd.DataFrame(records)
        df.to_csv(self.output_dir/"companies.csv", index=False)
        logger.info(f"Saved{len(df)} companies to src/data/raw/companies.csv")

        return df
    
def label_companies( companies: pd.DataFrame, disrupted_names: set[str],)-> pd.DataFrame:
    """
    add binary disrupted column to the companies df 
    match on uppercase name using substring containment
    """
    def is_disrupted(name:str)->int:
        name_up  = name.upper().strip()
        return int(any(d in name_up or name_up in d for d in disrupted_names))
    
    companies = companies.copy()
    companies["disrupted"] = companies["name"].apply(is_disrupted)
    pos = companies["disrupted"].sum()
    logger.info(
        f"Labelled {len(companies)} companies: "
        f" {pos} disrupted ({100 * pos / len(companies):.1f}%)"
    )
    return companies


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-companies", type=int, default=5000)
    parser.add_argument("--years", type=int, nargs="+", default=list(range(1995, 2026)))

    args = parser.parse_args()

    #SEC EDGAR
    sec = SECEdgarScraper()
    companies = sec.build_company_dataset(args.max_companies)

    #wikipedia chapter 11 labels
    label_scraper = DisruptionLabelScraper()
    disrupted_names = label_scraper.builld_disrupted_name_set(args.years)

    #label companies
    companies = label_companies(companies, disrupted_names)
    companies.to_csv("src/data/raw/companies_labelled.csv", index=False)

    print("\nTier distribution:")
    print(companies["tier"].value_counts().sort_index())
    print(f"\nDisruption rate: {companies['disrupted'].mean():.1%}")