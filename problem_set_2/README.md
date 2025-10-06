Excercise 1
```python
from hashlib import sha3_256, sha256, blake2b
from bitarray import bitarray
import matplotlib.pyplot as plt
import json
```

1a.
```python
class BloomFilter:
    def __init__(self, size, num_hashes=3):
        self.size = size
        self.num_hashes = num_hashes
        self.bit_array = bitarray(size)
        self.bit_array.setall(0)
    
    def my_hash(self, s):
        return int(sha256(s.lower().encode()).hexdigest(), 16) % self.size
    
    def my_hash2(self, s):
        return int(blake2b(s.lower().encode()).hexdigest(), 16) % self.size
    
    def my_hash3(self, s):
        return int(sha3_256(s.lower().encode()).hexdigest(), 16) % self.size
    
    def add(self, word):
        if self.num_hashes >= 1:
            self.bit_array[self.my_hash(word)] = 1
        if self.num_hashes >= 2:
            self.bit_array[self.my_hash2(word)] = 1
        if self.num_hashes >= 3:
            self.bit_array[self.my_hash3(word)] = 1
    
    def contains(self, word):
        if self.num_hashes >= 1 and not self.bit_array[self.my_hash(word)]:
            return False
        if self.num_hashes >= 2 and not self.bit_array[self.my_hash2(word)]:
            return False
        if self.num_hashes >= 3 and not self.bit_array[self.my_hash3(word)]:
            return False
        return True
```
```python
# Populate Bloom Filter
bf = BloomFilter(10**7)
with open('words.txt') as f:
    for line in f:
        bf.add(line.strip())
```

1b.
```python
def spell_check(word, bloom_filter):
    suggestions = []
    for i in range(len(word)):
        for char in 'abcdefghijklmnopqrstuvwxyz':
            candidate = word[:i] + char + word[i+1:]
            if bloom_filter.contains(candidate):
                suggestions.append(candidate)
    return suggestions

# Test
print(spell_check('floeer', bf))
```
```python
# Evaluate Performance
data = json.load(open('typos.json'))
good = sum(1 for item in data if item[0] != item[1] and 
           len(s := spell_check(item[0], bf)) <= 3 and item[1] in s)
total = sum(1 for item in data if item[0] != item[1])
print(f"Good suggestions: {good/total*100:.2f}%")
```

1c.
```python
filter_sizes = [10**2, 10**3, 10**4, 10**5, 10**6, 10**7, 10**8]
results = {h: {'misidentified': [], 'good_suggestions': []} for h in [1,2,3]}

for num_hashes in [1, 2, 3]:
    for size in filter_sizes:
        bf_temp = BloomFilter(size, num_hashes)
        with open('words.txt') as f:
            for line in f:
                bf_temp.add(line.strip())
        
        data = json.load(open('typos.json'))
        typos = [item for item in data if item[0] != item[1]]
        
        misid = sum(1 for item in typos if item[1] not in spell_check(item[0], bf_temp))
        good = sum(1 for item in typos if len(s := spell_check(item[0], bf_temp)) <= 3 and item[1] in s)
        
        results[num_hashes]['misidentified'].append(misid/len(typos)*100)
        results[num_hashes]['good_suggestions'].append(good/len(typos)*100)
```
```python
# Plot
plt.figure(figsize=(10, 6))
colors = {1: ['blue', 'orange'], 2: ['green', 'red'], 3: ['purple', 'brown']}
for h in [1, 2, 3]:
    plt.plot(filter_sizes, results[h]['misidentified'], label=f'Misidentified %, {h} hashes', color=colors[h][0])
    plt.plot(filter_sizes, results[h]['good_suggestions'], label=f'good suggestion %, {h} hashes', color=colors[h][1])
plt.xscale('log')
plt.xlabel('Bits in Bloom Filter')
plt.ylabel('Percentage')
plt.ylim(0, 100)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```
```python
# Find 85% threshold
print("\nBits needed for 85% good suggestions:")
for h in [1, 2, 3]:
    for i, pct in enumerate(results[h]['good_suggestions']):
        if pct >= 85:
            print(f"{h} hash function(s): ~{filter_sizes[i]:.0e} bits")
            break
    else:
        print(f"{h} hash function(s): >10^8 bits needed")
```
