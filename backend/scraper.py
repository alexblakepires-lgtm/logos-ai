import requests
import time
from pathlib import Path
from bs4 import BeautifulSoup

BASE_URL = "https://www.materiamedica.info"
INDEX_URL = f"{BASE_URL}/en/materia-medica/james-tyler-kent/index"
OUTPUT_FILE = Path("../data/kent_materia_medica.txt")

def get_remedy_links():
    print("📖 Fetching index page...")
    response = requests.get(INDEX_URL)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    links = []
    for a in soup.find_all('a', href=True):
        href = a['href']
        if '/en/materia-medica/james-tyler-kent/' in href and 'index' not in href:
            full_url = BASE_URL + href if href.startswith('/') else href
            if full_url not in links:
                links.append(full_url)
    
    print(f"✅ Found {len(links)} remedy pages")
    return links

def scrape_remedy(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        title = soup.find('h1')
        name = title.get_text(strip=True) if title else url.split('/')[-1]
        
        paragraphs = soup.find_all('p')
        text = '\n'.join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
        
        return f"\n\n{'='*60}\n{name}\n{'='*60}\n{text}"
    except Exception as e:
        print(f"⚠️  Error scraping {url}: {e}")
        return ""

def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    links = get_remedy_links()
    
    all_text = "KENT'S MATERIA MEDICA - James Tyler Kent (1905)\n"
    all_text += "Public Domain - Originally published 1905\n\n"
    
    for i, url in enumerate(links):
        remedy_name = url.split('/')[-1].replace('-', ' ')
        print(f"[{i+1}/{len(links)}] Scraping: {remedy_name}")
        text = scrape_remedy(url)
        all_text += text
        time.sleep(0.5)
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(all_text)
    
    print(f"\n✅ Done! Saved to {OUTPUT_FILE}")
    print(f"   Total size: {len(all_text)} characters")

if __name__ == "__main__":
    main()