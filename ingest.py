from scraper import LegalScraper
from retrieval_engine import LegalRetrievalEngine

scraper = LegalScraper()
db = LegalRetrievalEngine()

# 1. Scrape the data
raw_data = scraper.fetch_and_parse("https://your-target-legal-site.com/act", act_name="The Act Name")

# 2. Inject it straight into Qdrant
if raw_data:
    db.add_legal_chunks(raw_data)
    print("Database updated! The bot is now smarter.")