Excercise 1

1a.
```python
import requests
import time
import xml.etree.ElementTree as ET
```
```python
# Fetch Alzheimer's papers
url_alzheimer = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=Alzheimers+AND+2024[pdat]&retmax=1000&retmode=xml"
response_alzheimer = requests.get(url_alzheimer)
time.sleep(1)

# Fetch Cancer papers
url_cancer = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=cancer+AND+2024[pdat]&retmax=1000&retmode=xml"
response_cancer = requests.get(url_cancer)

# Parse PubMed IDs
alzheimer_tree = ET.fromstring(response_alzheimer.content)
alzheimer_ids = [id_elem.text for id_elem in alzheimer_tree.findall('.//Id')]

cancer_tree = ET.fromstring(response_cancer.content)
cancer_ids = [id_elem.text for id_elem in cancer_tree.findall('.//Id')]
```

1b.
```python
import json

# Process in batches
all_ids = alzheimer_ids + cancer_ids
batch_size = 200
papers_metadata = {}

for i in range(0, len(all_ids), batch_size):
    batch_ids = all_ids[i:i+batch_size]
    id_string = ",".join(batch_ids)
    
    url_fetch = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={id_string}&retmode=xml"
    response_metadata = requests.get(url_fetch)
    
    metadata_tree = ET.fromstring(response_metadata.content)
    
    for article in metadata_tree.findall('.//PubmedArticle'):
        pmid_elem = article.find('.//PMID')
        if pmid_elem is not None:
            pmid = pmid_elem.text
            
            title_elem = article.find('.//ArticleTitle')
            title = ET.tostring(title_elem, encoding='unicode', method='text') if title_elem is not None else ""
            
            abstract_parts = []
            for abstract_text in article.findall('.//AbstractText'):
                text = ET.tostring(abstract_text, encoding='unicode', method='text')
                abstract_parts.append(text)
            abstract = " ".join(abstract_parts) if abstract_parts else ""
            
            query = "Alzheimer" if pmid in alzheimer_ids else "cancer"
            
            papers_metadata[pmid] = {
                "ArticleTitle": title,
                "AbstractText": abstract,
                "query": query
            }
    
    time.sleep(1)
```
```python
# Save to JSON
with open('papers_metadata.json', 'w', encoding='utf-8') as f:
    json.dump(papers_metadata, f, indent=2, ensure_ascii=False)
```

1c.
```python
alzheimer_set = set(alzheimer_ids)
cancer_set = set(cancer_ids)
overlap = alzheimer_set.intersection(cancer_set)

print(f"Number of overlapping papers: {len(overlap)}")
print(f"Overlapping PubMed IDs: {overlap}")
```
Number of overlapping papers: 4

Overlapping PubMed IDs: {'40395755', '40326981', '40949928', '40800467'}


1d.
Current method in 1b could ensure all sections are included. abstract is stored as one continuous string rather than a structured format (like a dictionary with separate keys for each section. For example for PMID:20966393, we can see subtitle rationale, objectives, etc. as the beginning of each paragraph, but the code stored all sections as a chunk.


Excercise 2
