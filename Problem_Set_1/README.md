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


1b
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
```
![Gender Distribution](gender_distribution.png)
