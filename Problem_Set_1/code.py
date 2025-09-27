# Exercise 1
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from collections import Counter

# 1a.
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
plt.show()

age_counts = Counter(ages)
duplicates = {age: count for age, count in age_counts.items() if count > 1}
print(f"Patients with same age: {duplicates if duplicates else 'None'}")


# 1b.
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


# 1c.
sorted_patients = sorted(patients, key=lambda x: x['age'])
print(f"Oldest patient: {sorted_patients[-1]['name']}, Age: {sorted_patients[-1]['age']}")


# 1d.
max_age = max(ages)
second_max = max(age for age in ages if age < max_age)
second_oldest = next(p for p in patients if p['age'] == second_max)
print(f"Second oldest: {second_oldest['name']}, Age: {second_oldest['age']}")


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


1f.
target_age = 41.5
idx = binary_search(sorted_patients, target_age)

count = len(sorted_patients) - idx
print(f"Patients >= {target_age} years old: {count}")


1g.
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
tests results: [30, 40): 43189 [40, 50): 42525 [25, 35): 42857

1h.

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



# Excercise 2
def administer_meds(delta_t, tstop):
    t = 0
    while t < tstop:
        print(f"Administering meds at t={t}")
        t += delta_t


# 2b.
administer_meds(0.25, 1)


# 2c.
administer_meds(0.1, 1)


# 2f.
def administer_meds_revised(delta_t, tstop):
    import math
    num_doses = int(math.ceil(tstop / delta_t))
    for i in range(num_doses):
        t = i * delta_t
        if t >= tstop:
            break
        print(f"Administering meds at t={t:.2f}")


        
# Exercise 3
import numpy as np
import time
import matplotlib.pyplot as plt
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


# 3a.
test_data = [5, 2, 8, 1, 9]
print(f"alg1 output: {alg1(test_data)}")
print(f"alg2 output: {alg2(test_data)}")


# 3c.
n_values = np.logspace(1, 4, 20).astype(int)

def measure_performance(alg, data_func, n_values, name):
    times = []
    for n in n_values:
        data = data_func(n)
        start = time.perf_counter()
        alg(data)
        end = time.perf_counter()
        times.append(end - start)
    return times

# Data set 1
times_alg1_data1 = measure_performance(alg1, data1, n_values, "alg1-data1")
times_alg2_data1 = measure_performance(alg2, data1, n_values, "alg2-data1")

# Data set 2
times_alg1_data2 = measure_performance(alg1, data2, n_values, "alg1-data2")
times_alg2_data2 = measure_performance(alg2, data2, n_values, "alg2-data2")

# Data set 3
times_alg1_data3 = measure_performance(alg1, data3, n_values, "alg1-data3")
times_alg2_data3 = measure_performance(alg2, data3, n_values, "alg2-data3")

# Plot data1
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.loglog(n_values, times_alg1_data1, 'o-', label='alg1')
plt.loglog(n_values, times_alg2_data1, 's-', label='alg2')
plt.xlabel('n')
plt.ylabel('Time (s)')
plt.title('Performance on data1')
plt.legend()
plt.grid(True)

# Plot data2
plt.subplot(1, 3, 2)
plt.loglog(n_values, times_alg1_data2, 'o-', label='alg1')
plt.loglog(n_values, times_alg2_data2, 's-', label='alg2')
plt.xlabel('n')
plt.ylabel('Time (s)')
plt.title('Performance on data2')
plt.legend()
plt.grid(True)

# Plot data3
plt.subplot(1, 3, 3)
plt.loglog(n_values, times_alg1_data3, 'o-', label='alg1')
plt.loglog(n_values, times_alg2_data3, 's-', label='alg2')
plt.xlabel('n')
plt.ylabel('Time (s)')
plt.title('Performance on data3')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()



# Exercise 4
import time
import random
import matplotlib.pyplot as plt
import numpy as np

# 4abc.
class Tree:
    def __init__(self):
        self._value = None
        self._data = None
        self.left = None
        self.right = None

    # 4a. 
    def add(self, value, data):
        if self._value is None:
            self._value = value
            self._data = data
        elif value < self._value:
            if self.left is None:
                self.left = Tree()
            self.left.add(value, data)
        else:
            if self.right is None:
                self.right = Tree()
            self.right.add(value, data)

    # 4b. 
    def __contains__(self, patient_id):
        if self._value == patient_id:
            return True
        elif self.left and patient_id < self._value:
            return patient_id in self.left
        elif self.right and patient_id > self._value:
            return patient_id in self.right
        else:
            return False

    # 4c.
    def has_data(self, data):
        if self._data == data:
            return True
        if self.left and self.left.has_data(data):
            return True
        if self.right and self.right.has_data(data):
            return True
        return False

# Test 4a and 4b
print("4a & 4b Tests:")
my_tree = Tree()
for patient_id, initials in [(24601, "JV"), (42, "DA"), (7, "JB"), (143, "FR"), (8675309, "JNY")]:
    my_tree.add(patient_id, initials)

print(f"24601 in my_tree: {24601 in my_tree}")
print(f"1492 in my_tree: {1492 in my_tree}")

# Test 4c
print("4c Tests:")
print(f"has_data('JV'): {my_tree.has_data('JV')}")
print(f"has_data(24601): {my_tree.has_data(24601)}")


# 4c.
n_values = np.logspace(2, 4, 15).astype(int)
contains_times = []
has_data_times = []
setup_times = []

for n in n_values:
    # Generate random patient data
    patient_ids = random.sample(range(1, n*10), n)
    patient_data = [f"P{i}" for i in range(n)]
    
    # Measure setup time
    tree = Tree()
    start = time.perf_counter()
    for pid, data in zip(patient_ids, patient_data):
        tree.add(pid, data)
    setup_time = time.perf_counter() - start
    setup_times.append(setup_time)
    
    # Measure __contains__ time
    test_ids = random.sample(patient_ids, min(1000, n))
    start = time.perf_counter()
    for tid in test_ids:
        _ = tid in tree
    contains_time = (time.perf_counter() - start) / len(test_ids)
    contains_times.append(contains_time)
    
    # Measure has_data time
    test_data = random.sample(patient_data, min(1000, n))
    start = time.perf_counter()
    for td in test_data:
        _ = tree.has_data(td)
    has_data_time = (time.perf_counter() - start) / len(test_data)
    has_data_times.append(has_data_time)

# Plot performance
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.loglog(n_values, contains_times, 'o-')
plt.xlabel('n')
plt.ylabel('Time per operation (s)')
plt.title('__contains__ Performance')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.loglog(n_values, has_data_times, 'o-')
plt.xlabel('n')
plt.ylabel('Time per operation (s)')
plt.title('has_data Performance')
plt.grid(True)

plt.subplot(1, 3, 3)
plt.loglog(n_values, setup_times, 'o-', label='Actual')
plt.loglog(n_values, n_values * 1e-6, '--', label='O(n)')
plt.loglog(n_values, n_values * np.log(n_values) * 1e-7, '--', label='O(n log n)')
plt.xlabel('n')
plt.ylabel('Total setup time (s)')
plt.title('Tree Construction Time')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
