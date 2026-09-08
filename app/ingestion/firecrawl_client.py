import os
from dotenv import load_dotenv
from firecrawl import Firecrawl

load_dotenv()

firecrawl=Firecrawl(
    api_key=os.getenv("FIRECRAWL_API_KEY")
)
def scrap_url(url:str):
    result=firecrawl.scrape(
        url,
        formats=['markdown']
    )
    return result