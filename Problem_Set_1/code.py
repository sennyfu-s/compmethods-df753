# Excercise 1
# improt the following before run the code of excercise 1
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from collections import Counter

# Load and parse XML
tree = ET.parse('pset1-patients.xml')
root = tree.getroot()

# Extract patient data
patients = []
for patient in root.findall('patient'):
    patients.append({
        'name': patient.find('name').text,
        'age': float(patient.find('age').text),
        'gender': patient.find('gender').text})


# 1a
ages = [p['age'] for p in patients]
plt.figure(figsize=(10, 6))
plt.hist(ages, bins=20, edgecolor='black')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Distribution of Patients')
plt.savefig('age_distribution.png')
plt.close()

age_counts = Counter(ages)
duplicates = {age: count for age, count in age_counts.items() if count > 1}
print(f"Patients with same age: {duplicates if duplicates else 'None'}")
