Name: Dongshi Senny Fu
NetID: df753

Question 1a:
Please run line 1-7 of prob_0.py

Question 1b:
The phrase 'within 1 degree' is ambiguous. it could include or exclude the boundary.

Question 1c:
Please run line 10-25 of prob_0.py
Test Results:
chicken_tester(42): True, #True -- i.e. not a fever for a chicken
human_tester(42): False, #False -- this would be a severe fever for a human
chicken_tester(43): False, #False
human_tester(35): False, #False -- too low
human_tester(98.6): False, #False -- normal in degrees F but our reference temp was in degrees C



Excercise 2
pandas and matplotlib need to be installed for this question: pip install pandas matplotlib
Please run line 329-33 of prob_0.py to import the packages

Question 2a:
Data Source Credit: The New York Times COVID-19 Data Repository
Repository: github.com/nytimes/covid-19-data
Please run line 36-38 of prob_0.py

Question 2b:
Please run line 46-72 of prob_0.py to generate the plotting function
Limitation: it records negative number for 'new cases'.

Question 2c:
Please run line 80-87 of prob_0.py to generate the plotting function
Example of the function use:
state = 'New York'  # Select the city of interest
peak_date = find_peak_date(state)  # run the function
peak_cases = data[(data['state'] == state) & (data['date'] == peak_date)]['new_cases'].iloc[0]  # Extract the information
print(f"{state}: {peak_date.strftime('%Y-%m-%d')} ({int(peak_cases):,} new cases)")  # Display the information
The result looks like:
New York: 2022-01-08 (90,132 new cases)

Question 2d:
Please run line 96-116 of prob_0.py to generate the plotting function
Example of the function use:
first_state, days_diff = compare_peak_cases('New York', 'California')  # Change the city name with cities interested
print("New York vs California:")
print(f"{first_state} peaked first")
print(f"Days between peaks: {days_diff}")
The result looks like:
New York vs California:
New York peaked first
Days between peaks: 2

Question 2e:
Please run line 128 of prob_0.py to generate the plot for Florida
Observation: There's a sharp peak occuring in between 2021-11 and 2022-02. This massive January 2022 spike in Florida's COVID cases likely resulted from both New Year's social gatherings driving real transmission and holiday reporting delays.


Question 3:
Please install and import the packages in line 133-136 before running the code

Question 3a:
The data contians column 'name', 'age', 'weight', and 'eye color'.
There are 152,361 individuals recorded in this data set.

Question 3b:
Mean: 39.51
Std: 24.15
Min: 0.00
Max: 99.99
Please run line 151-155 for the age histogram.
Justification of bin selection: 20 bins provides a good balance between showing the overall distribution shape while maintaining enough detail to identify key features.
Outlier comment: There's an obvious gap in the data around ages 60-80, the data may be bias and could not reflect the true pattern of the older generation.

Question 3c:
Mean: 60.88
Std: 18.41
Min: 3.38
Max: 100.44
Question 3c:
Please run line 167-171 for the weight histogram.
Outlier comment: There's a small but noticeable frequency of very low weights that are unrealistic for human populations (ie. weights ~0-5). This data will be misleading when doing downing stream analysis, e.g. fitting model.

Question 3d:
Please run line 174-180 for the plot.
General relationship: A moderate positive correlation (r = 0.640) between weight and age. As people age from childhood through middle age, their weight increases, but this trend levels off after reaching adult stage (~age 20).
Outlier: Anthony Freeman, Age 41.3, Weight 21.7.
Process for identifying outliers: The outlier was identified through direct visual inspection of the scatterplot. Anthony Freeman (weight 21.7) appears as an isolated point far below the main data cluster around age 40 (weight ~50-90).


Question 4:
Please import the packages and files by running line 194-199

Question 4a:
More common gender: F
M: 45, F: 55

Question 4b:
Please run line 208-217 to define the function.

Question 4c:
Please run line 221-222 to see the test results.
The function successfully links all three tables and correctly picks out the patients with trait of interest.

Question 4d:
Patient 10043: 29891 days (81.9 years)
Patient 10045: 25087 days (68.7 years)
Patient 10094: 109593 days (300.3 years)  # unrealistic age
Patient 10102: 25626 days (70.2 years)
Patient 40595: 27999 days (76.7 years)
Patient 41976: 24235 days (66.4 years)
Patient 44228: 21358 days (58.5 years)

Question 4e:
Working with data split across multiple tables required understanding the relationships between datasets and implementing proper joins to link separate files using foreign keys (subject_id and icd9_code). This was more complex than single-table analysis but prevent data duplication and give better organization information storage.
Alternative data representations such as a dictionary keyed by subject_id containing all patient information and diagnoses bundled under each key. Compared to the tabular structure, dictionary formats offer faster direct access and group related data together but suffer from data duplication and maintenance challenges.
To transform from tabular to hierarchical format, code would iterate through each patient, look up their diagnoses by linking the tables, convert ICD codes to descriptions, and group all information under the patient's key.
