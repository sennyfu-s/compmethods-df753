Excercise 1
Please import the following before run the code.
```python
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from collections import Counter
```

1a.
```python
# Load and parse XML
tree = ET.parse('pset1-patients.xml')
root = tree.getroot()

# Extract patient data
patients = []
patients_node = root.find('patients')

for patient in patients_node.findall('patient'):
    patients.append({
        'name': patient.get('name', 'Unknown'),
        'age': float(patient.get('age', 0)),
        'gender': patient.get('gender', 'Unknown')
    })

ages = [p['age'] for p in patients]
plt.figure(figsize=(10, 6))
plt.hist(ages, bins=20, edgecolor='black')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Distribution of Patients')
plt.savefig('age_distribution.png')
plt.close()
```

![Age Distribution](age_distribution.png)

There is no patient share the same exact age. Because the following code returns 'None'.
```python
age_counts = Counter(ages)
duplicates = {age: count for age, count in age_counts.items() if count > 1}
print(f"Patients with same age: {duplicates if duplicates else 'None'}")
```
Extra Credit: When multiple patients have the same age, binary search may return any of them, requiring additional logic to find all matches


1b.
```python
genders = [p['gender'] for p in patients]
gender_counts = Counter(genders)
total = len(genders)
gender_percentages = {k: (v/total)*100 for k, v in gender_counts.items()}

plt.figure(figsize=(8, 6))
plt.bar(gender_percentages.keys(), gender_percentages.values())
plt.xlabel('Gender')
plt.ylabel('Percentage (%)')
plt.title('Gender Distribution')
plt.savefig('gender_distribution.png')
plt.close()

print(f"Gender encoding: {dict(gender_counts)}")
print(f"Gender percentages: {gender_percentages}")
```
![Gender Distribution](gender_distribution.png)
Gender Encoding: strings ('female', 'male', 'unknown')
Categories:
female: 165,293 (50.96%)
male: 158,992 (49.02%)
unknown: 72 (0.02%)


1c.
```python
sorted_patients = sorted(patients, key=lambda x: x['age'])
print(f"Oldest patient: {sorted_patients[-1]['name']}, Age: {sorted_patients[-1]['age']}")
```
Oldest patient: Monica Caponera, Age: 84.99855742449432


1d.
```python
# 1d.
max_age = max(ages)
second_max = max(age for age in ages if age < max_age)
second_oldest = next(p for p in patients if p['age'] == second_max)
print(f"Second oldest: {second_oldest['name']}, Age: {second_oldest['age']}")
```
Second oldest: Raymond Leigh, Age: 84.9982928781625
Sorting is advantageous when Multiple queries. If need to find the kth oldest patient multiple times (2nd, 3rd, 4th oldest, etc.), sorting once O(n log n) + accessing O(1) is better than running O(n) each time.


1e.
```python
# 1e.
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid]['age'] == target:
            return mid
        elif arr[mid]['age'] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

target_age = 41.5
idx = binary_search(sorted_patients, target_age)
print(f"Patient aged 41.5: {sorted_patients[idx]['name']}")
```
Patient aged 41.5: John Braswell


1f.
```python
target_age = 41.5
idx = binary_search(sorted_patients, target_age)

count = len(sorted_patients) - idx
print(f"Patients >= {target_age} years old: {count}")
```
Patients >= 41.5 years old: 150471


1g.
```python
def count_age_range(low_age, high_age):
    # Binary search for boundaries in sorted ages
    sorted_ages = sorted(ages)
    
    # Find first index >= low_age
    left = 0
    right = len(sorted_ages)
    while left < right:
        mid = (left + right) // 2
        if sorted_ages[mid] < low_age:
            left = mid + 1
        else:
            right = mid
    start_idx = left
    
    # Find first index >= high_age
    left = 0
    right = len(sorted_ages)
    while left < right:
        mid = (left + right) // 2
        if sorted_ages[mid] < high_age:
            left = mid + 1
        else:
            right = mid
    end_idx = left
    
    return end_idx - start_idx

# Test cases
print(f"tests results:")
print(f"[30, 40): {count_age_range(30, 40)}")
print(f"[40, 50): {count_age_range(40, 50)}")
print(f"[25, 35): {count_age_range(25, 35)}")
```
tests results:
[30, 40): 43189
[40, 50): 42525
[25, 35): 42857


1h.
```python
# Pre-sort gender-specific lists
male_ages = sorted([p['age'] for p in patients if p['gender'] == 'male'])
female_ages = sorted([p['age'] for p in patients if p['gender'] == 'female'])

def count_age_gender_range(low_age, high_age, gender='male'):
    # Select pre-sorted list based on gender
    if gender == 'male' or gender == 'M':
        gender_ages = male_ages
    elif gender == 'female' or gender == 'F':
        gender_ages = female_ages
    else:
        return 0
    
    # Binary search for low_age boundary
    left, right = 0, len(gender_ages)
    while left < right:
        mid = (left + right) // 2
        if gender_ages[mid] < low_age:
            left = mid + 1
        else:
            right = mid
    start_idx = left
    
    # Binary search for high_age boundary
    left, right = 0, len(gender_ages)
    while left < right:
        mid = (left + right) // 2
        if gender_ages[mid] < high_age:
            left = mid + 1
        else:
            right = mid
    end_idx = left
    
    return end_idx - start_idx

# Test cases
print(f"test results:")
print(f"Males [30, 40): {count_age_gender_range(30, 40, 'male')}")
print(f"Females [30, 40): {count_age_gender_range(30, 40, 'female')}")
print(f"Males [40, 50): {count_age_gender_range(40, 50, 'male')}")
print(f"Females [40, 50): {count_age_gender_range(40, 50, 'female')}")
```
test results:
Males [30, 40): 21606
Females [30, 40): 21574
Males [40, 50): 20873
Females [40, 50): 21641



Excercise 2
2a.
The function administers medication at regular time intervals (delta_t) until the total time tstop is reached. It starts at t=0, prints administration messages, and increments time by delta_t each iteration.
tstop: Total duration of treatment
delta_t: Time interval between doses
Expected doses: tstop / delta_t


2b.
```python
def administer_meds(delta_t, tstop):
    t = 0
    while t < tstop:
        print(f"Administering meds at t={t}")
        t += delta_t
```
```python
administer_meds(0.25, 1)
```
Administering meds at t=0
Administering meds at t=0.25
Administering meds at t=0.5
Administering meds at t=0.75


2c.
```python
administer_meds(0.1, 1)
```
Administering meds at t=0
Administering meds at t=0.1
Administering meds at t=0.2
Administering meds at t=0.30000000000000004
Administering meds at t=0.4
Administering meds at t=0.5
Administering meds at t=0.6
Administering meds at t=0.7
Administering meds at t=0.7999999999999999
Administering meds at t=0.8999999999999999
Administering meds at t=0.9999999999999999


2d.
2b: Got exactly 4 doses as expected
2c: Got 11 doses instead of expected 10
Explain: Floating-point representation error. In binary, 0.1 cannot be represented exactly, causing accumulation errors.


2e.
Extra doses can cause toxicity and adverse reactions.


2f.
```python
def administer_meds_revised(delta_t, tstop):
    import math
    num_doses = int(math.ceil(tstop / delta_t))
    for i in range(num_doses):
        t = i * delta_t
        if t >= tstop:
            break
        print(f"Administering meds at t={t:.2f}")
```
My function eliminates floating-point accumulation errors by using integer arithmetic for iteration and calculating time values directly.



Excercise 3
Import the following before run the code.
```python
import numpy as np
import time
import matplotlib.pyplot as plt
```
```python
def alg1(data):
  data = list(data)
  changes = True
  while changes:
    changes = False
    for i in range(len(data) - 1):
      if data[i + 1] < data[i]:
        data[i], data[i + 1] = data[i + 1], data[i]
        changes = True
  return data

def alg2(data):
  if len(data) <= 1:
    return data
  else:
    split = len(data) // 2
    left = iter(alg2(data[:split]))
    right = iter(alg2(data[split:]))
    result = []
    # note: this takes the top items off the left and right piles
    left_top = next(left)
    right_top = next(right)
    while True:
      if left_top < right_top:
        result.append(left_top)
        try:
          left_top = next(left)
        except StopIteration:
          # nothing remains on the left; add the right + return
          return result + [right_top] + list(right)
      else:
        result.append(right_top)
        try:
          right_top = next(right)
        except StopIteration:
          # nothing remains on the right; add the left + return
          return result + [left_top] + list(left)

def data1(n, sigma=10, rho=28, beta=8/3, dt=0.01, x=1, y=1, z=1):
    import numpy
    state = numpy.array([x, y, z], dtype=float)
    result = []
    for _ in range(n):
        x, y, z = state
        state += dt * numpy.array([
            sigma * (y - x),
            x * (rho - z) - y,
            x * y - beta * z
        ])
        result.append(float(state[0] + 30))
    return result

def data2(n):
    return list(range(n))

def data3(n):
    return list(range(n, 0, -1))
```


3a.
```python
test_data = [5, 2, 8, 1, 9]
print(f"alg1 output: {alg1(test_data)}")
print(f"alg2 output: {alg2(test_data)}")
```
alg1 output: [1, 2, 5, 8, 9]
alg2 output: [1, 2, 5, 8, 9]
Hypothesis: Both algorithms perform sorting in ascending order.


3b.
alg1 implements bubble sort by repeatedly comparing adjacent elements and swapping them if out of order until no swaps are needed. While alg2 implements merge sort by recursively splitting the list in half, then merging sorted sublists back together in order.


3d.
alg1 is fastest on already-sorted data but becomes slow on random or reversed data as size increases. alg2 maintains consistent performance regardless of input order. I recommend to use alg2 for healthcare applications because its predictable O(n log n) performance ensures reliable response times regardless of data characteristics. Use alg1 only for small datasets (maybe ~n < 100) that are known to be nearly sorted.
