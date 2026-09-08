from firecrawl_client import scrap_url

url = "https://arxiv.org/abs/2005.11401"
result= scrap_url(url)
print(result.markdown)