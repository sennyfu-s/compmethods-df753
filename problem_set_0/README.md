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


Question 3a:
