Excercise 1

This is not a clean URL as it shows the .py extension.
```python
import requests

a, b = 0.5, 0.5
lr = 0.01
h = 1e-5

for i in range(1000):
    e = float(requests.get(f"http://ramcdougal.com/cgi-bin/error_function.py?a={a}&b={b}", 
                           headers={"User-Agent": "MyScript"}).text)
    ea = float(requests.get(f"http://ramcdougal.com/cgi-bin/error_function.py?a={a+h}&b={b}", 
                            headers={"User-Agent": "MyScript"}).text)
    eb = float(requests.get(f"http://ramcdougal.com/cgi-bin/error_function.py?a={a}&b={b+h}", 
                            headers={"User-Agent": "MyScript"}).text)
    
    grad_a = (ea - e) / h
    grad_b = (eb - e) / h
    
    a_new = a - lr * grad_a
    b_new = b - lr * grad_b
    
    if abs(a_new - a) < 1e-6 and abs(b_new - b) < 1e-6:
        break
    
    a, b = a_new, b_new

print(f"a={a:.3f}, b={b:.3f}, error={e:.3f}")
```
a=0.216, b=0.689, error=1.100

The gradient is estimated using numerical differentiation with finite differences. Since the error function is a black-box API, I approximate the partial derivative for each parameter by computing (f(x+h) - f(x))/h. For parameter a, I keep b constant and measure the error change when a increases by h. Similarly for b. This gives the gradient vector, then I subtract it to descend toward the minimum.

Learning rate (0.01): Small enough for stable convergence without overshooting.

Step size h (1e-5): Balances accurate derivative approximation with numerical stability in floating-point arithmetic.

Tolerance (1e-6): Stops when parameter changes are negligible.


Exercise 2
```python
import numpy as np

def align(seq1, seq2, match=1, gap_penalty=1, mismatch_penalty=1):
    m, n = len(seq1), len(seq2)
    H = np.zeros((m + 1, n + 1), dtype=int)
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            match_score = H[i-1, j-1] + (match if seq1[i-1] == seq2[j-1] else -mismatch_penalty)
            delete = H[i-1, j] - gap_penalty
            insert = H[i, j-1] - gap_penalty
            H[i, j] = max(0, match_score, delete, insert)
    
    # Find maximum score
    max_score = np.max(H)
    max_pos = np.unravel_index(np.argmax(H), H.shape)
    
    # Backtrack
    align1, align2 = '', ''
    i, j = max_pos
    while i > 0 and j > 0 and H[i, j] > 0:
        score = H[i, j]
        diag = H[i-1, j-1]
        up = H[i-1, j]
        left = H[i, j-1]
        
        if score == diag + (match if seq1[i-1] == seq2[j-1] else -mismatch_penalty):
            align1 = seq1[i-1] + align1
            align2 = seq2[j-1] + align2
            i -= 1
            j -= 1
        elif score == up - gap_penalty:
            align1 = seq1[i-1] + align1
            align2 = '-' + align2
            i -= 1
        else:
            align1 = '-' + align1
            align2 = seq2[j-1] + align2
            j -= 1
    
    return align1, align2, max_score
```
```python
# 1
seq1, seq2, score = align('tgcatcgagaccctacgtgac', 'actagacctagcatcgac')
print("Test 1 (default parameters):")
print(f"seq1 = {seq1}")
print(f"seq2 = {seq2}")
print(f"score = {score}\n")

# 2
seq1, seq2, score = align('tgcatcgagaccctacgtgac', 'actagacctagcatcgac', gap_penalty=2)
print("Test 2 (gap_penalty=2):")
print(f"seq1 = {seq1}")
print(f"seq2 = {seq2}")
print(f"score = {score}\n")

# 3
seq1, seq2, score = align('tgcatcgagaccctacgtgac', 'actagacctagcatcgac', match=2)
print("Test 3 (match=2):")
print(f"seq1 = {seq1}")
print(f"seq2 = {seq2}")
print(f"score = {score}\n")

# 4
seq1, seq2, score = align('tgcatcgagaccctacgtgac', 'actagacctagcatcgac', mismatch_penalty=2)
print("Test 4 (mismatch_penalty=2):")
print(f"seq1 = {seq1}")
print(f"seq2 = {seq2}")
print(f"score = {score}\n")

# 5
seq1, seq2, score = align('tgcatcgagaccctacgtgac', 'actagacctagcatcgac', match=2, gap_penalty=2, mismatch_penalty=2)
print("Test 5 (match=2, gap_penalty=2, mismatch_penalty=2):")
print(f"seq1 = {seq1}")
print(f"seq2 = {seq2}")
print(f"score = {score}")
```
Test 1 (default parameters):
seq1 = agacccta-cgt-gac
seq2 = aga-cctagcatcgac
score = 8

Test 2 (gap_penalty=2):
seq1 = gcatcga
seq2 = gcatcga
score = 7

Test 3 (match=2):
seq1 = atcgagacccta-cgt-gac
seq2 = a-ctaga-cctagcatcgac
score = 22

Test 4 (mismatch_penalty=2):
seq1 = gcatcga
seq2 = gcatcga
score = 7

Test 5 (match=2, gap_penalty=2, mismatch_penalty=2):
seq1 = agacccta-cgt-gac
seq2 = aga-cctagcatcgac
score = 16
