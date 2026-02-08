import json
import pandas as pd
from collections import Counter, defaultdict
from statistics import mean, median, stdev

def analyze_dataset(filepath):
    """Analyze the dataset.jsonl file for various statistics."""
    
    articles_per_year = defaultdict(int)
    codes_per_article_per_year = defaultdict(list)
    all_codes_per_article = []
    all_codes = []
    articles_per_year_list = defaultdict(list)
    code_major_status = defaultdict(list)
    
    total_articles = 0
    
    # Read and process the dataset
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                article = json.loads(line)
                year = article['pub_year']
                mesh_codes = article['mesh']
                
                articles_per_year[year] += 1
                total_articles += 1
                num_codes = len(mesh_codes)
                all_codes_per_article.append(num_codes)
                codes_per_article_per_year[year].append(num_codes)
                
                # Extract code information
                for code in mesh_codes:
                    all_codes.append(code['name'])
                    code_major_status[code['name']].append(code['major'])
                    articles_per_year_list[year].append(code['name'])
    
    # Compute statistics
    print("=" * 80)
    print("DATASET ANALYSIS")
    print("=" * 80)
    print(f"\nTotal articles: {total_articles}")
    print(f"Year range: {min(articles_per_year.keys())} - {max(articles_per_year.keys())}")
    
    print("\n" + "=" * 80)
    print("ARTICLES PER YEAR")
    print("=" * 80)
    for year in sorted(articles_per_year.keys()):
        print(f"  {year}: {articles_per_year[year]:5d} articles")
    
    print("\n" + "=" * 80)
    print("CODES PER ARTICLE STATISTICS")
    print("=" * 80)
    print(f"\nOverall:")
    print(f"  Total codes: {len(all_codes)}")
    print(f"  Unique codes: {len(set(all_codes))}")
    print(f"  Average codes per article: {mean(all_codes_per_article):.2f}")
    print(f"  Median codes per article: {median(all_codes_per_article):.2f}")
    print(f"  Min codes per article: {min(all_codes_per_article)}")
    print(f"  Max codes per article: {max(all_codes_per_article)}")
    if len(all_codes_per_article) > 1:
        print(f"  Std dev codes per article: {stdev(all_codes_per_article):.2f}")
    
    print(f"\nPer Year:")
    for year in sorted(codes_per_article_per_year.keys()):
        codes = codes_per_article_per_year[year]
        print(f"  {year}:")
        print(f"    Average: {mean(codes):.2f}")
        print(f"    Median: {median(codes):.2f}")
        print(f"    Min/Max: {min(codes)}/{max(codes)}")
        if len(codes) > 1:
            print(f"    Std dev: {stdev(codes):.2f}")
    
    print("\n" + "=" * 80)
    print("TOP 20 MOST COMMON MESH CODES")
    print("=" * 80)
    code_counts = Counter(all_codes)
    for i, (code, count) in enumerate(code_counts.most_common(20), 1):
        percentage = (count / len(all_codes)) * 100
        print(f"  {i:2d}. {code:50s} {count:5d} ({percentage:5.2f}%)")
    
    print("\n" + "=" * 80)
    print("MAJOR VS MINOR CODES")
    print("=" * 80)
    total_major = sum(1 for statuses in code_major_status.values() for is_major in statuses if is_major)
    total_minor = len(all_codes) - total_major
    print(f"  Total major codes: {total_major} ({(total_major/len(all_codes)*100):.2f}%)")
    print(f"  Total minor codes: {total_minor} ({(total_minor/len(all_codes)*100):.2f}%)")
    
    print("\n" + "=" * 80)
    print("UNIQUE CODES PER YEAR")
    print("=" * 80)
    for year in sorted(articles_per_year_list.keys()):
        unique_codes = len(set(articles_per_year_list[year]))
        print(f"  {year}: {unique_codes:4d} unique codes")
    
    print("\n" + "=" * 80)
    print("ARTICLE DISTRIBUTION")
    print("=" * 80)
    code_freq = Counter(all_codes_per_article)
    for num_codes in sorted(code_freq.keys())[:15]:
        count = code_freq[num_codes]
        bar = "█" * (count // 20 + 1) if count > 0 else ""
        print(f"  {num_codes:3d} codes: {count:5d} articles {bar}")
    
    print("\n")

if __name__ == "__main__":
    analyze_dataset("dataset.jsonl")
