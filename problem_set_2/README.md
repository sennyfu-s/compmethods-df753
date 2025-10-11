Excercise 1
1a.
```python
from bitarray import bitarray
from hashlib import sha3_256, sha256, blake2b
```
```python
# Create Bloom filter
size = 10**7  # used 1 million bits before but the outcome is not ideal
bloom_filter = bitarray(size)
bloom_filter.setall(0)

# Define hash functions
def my_hash1(s):
    return int(sha256(s.lower().encode()).hexdigest(), 16) % size

def my_hash2(s):
    return int(blake2b(s.lower().encode()).hexdigest(), 16) % size

def my_hash3(s):
    return int(sha3_256(s.lower().encode()).hexdigest(), 16) % size
```
```python
# Insert words into Bloom filter
with open('words.txt') as f:
    for line in f:
        word = line.strip()
        if word:
            bloom_filter[my_hash1(word)] = 1
            bloom_filter[my_hash2(word)] = 1
            bloom_filter[my_hash3(word)] = 1
```

1b.
```python
def suggest_corrections(typo):
    suggestions = []
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    
    for i in range(len(typo)):
        for char in alphabet:
            candidate = typo[:i] + char + typo[i+1:]
            
            # Check if candidate is in Bloom filter
            if (bloom_filter[my_hash1(candidate)] and 
                bloom_filter[my_hash2(candidate)] and 
                bloom_filter[my_hash3(candidate)]):
                suggestions.append(candidate)
    
    return suggestions
```
```python
import json
```
```python
# Load typos dataset
with open('typos.json') as f:
    typos_data = json.load(f)
```
```python
def evaluate_bloom_filter(typos_data):
    good_suggestions = 0
    total = len(typos_data)
    
    for typo, correct in typos_data:
        suggestions = suggest_corrections(typo)
        
        if len(suggestions) <= 3 and correct in suggestions:
            good_suggestions += 1
    
    accuracy = (good_suggestions / total) * 100
    print(f"Good suggestions: {good_suggestions}/{total} ({accuracy:.2f}%)")
    return accuracy
```
```python
# Run evaluation
evaluate_bloom_filter(typos_data)
```
Result: 
Bloom filter created with size: 10000000
Good suggestions: 22907/50000 (45.81%) (I've tried with higher bit, ie 10^8, but it only increases to ~47%)

1c.
```python
import matplotlib.pyplot as plt
```
```python
def experiment_bloom_filter(num_hashes, filter_size):
    bf = bitarray(filter_size)
    bf.setall(0)
    
    # Define hash functions
    def hash1(s):
        return int(sha256(s.lower().encode()).hexdigest(), 16) % filter_size
    
    def hash2(s):
        return int(blake2b(s.lower().encode()).hexdigest(), 16) % filter_size
    
    def hash3(s):
        return int(sha3_256(s.lower().encode()).hexdigest(), 16) % filter_size
    
    hash_funcs = [hash1, hash2, hash3][:num_hashes]
    
    # Insert words
    with open('words.txt') as f:
        for line in f:
            word = line.strip()
            if word:
                for hf in hash_funcs:
                    bf[hf(word)] = 1
    
    # Evaluate
    good = 0
    total_fps = 0
    total_checks = 0
    
    for typo, correct in typos_data:
        suggestions = []
        
        for i in range(len(typo)):
            for char in 'abcdefghijklmnopqrstuvwxyz':
                candidate = typo[:i] + char + typo[i+1:]
                total_checks += 1
                
                if all(bf[hf(candidate)] for hf in hash_funcs):
                    suggestions.append(candidate)
                    if candidate != correct:
                        total_fps += 1
        
        if len(suggestions) <= 3 and correct in suggestions:
            good += 1
    
    mis_rate = (total_fps / total_checks) * 100
    good_rate = (good / len(typos_data)) * 100
    
    return mis_rate, good_rate
```
```python
# Plot
sizes = [10**i for i in range(3, 9)]
hash_counts = [1, 2, 3]

plt.figure(figsize=(10, 6))

for num_hashes in hash_counts:
    mis_rates = []
    good_rates = []
    
    for size in sizes:
        mis, good = experiment_bloom_filter(num_hashes, size)
        mis_rates.append(mis)
        good_rates.append(good)
    
    plt.plot(sizes, mis_rates, label=f'Misidentified %, {num_hashes} hashes')
    plt.plot(sizes, good_rates, label=f'good suggestion %, {num_hashes} hashes')

plt.xscale('log')
plt.xlabel('Bits in Bloom Filter')
plt.ylabel('Percentage')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("Performance_analysis_of_hashes.png")
plt.show()
```
![Bloom Filter Performance](Perfomance_analysis_of_hashes.png)
