# Question 1a:
def temp_tester(normal_temp):

    def tester(temperature):
        return abs(temperature - normal_temp) <= 1
    
    return tester


# Question 1c:
human_tester = temp_tester(37)
chicken_tester = temp_tester(41.1)


print("Test Results:")
test_cases = [
    ("chicken_tester(42)", chicken_tester(42), "True -- i.e. not a fever for a chicken"),
    ("human_tester(42)", human_tester(42), "False -- this would be a severe fever for a human"),
    ("chicken_tester(43)", chicken_tester(43), "False"),
    ("human_tester(35)", human_tester(35), "False -- too low"),
    ("human_tester(98.6)", human_tester(98.6), "False -- normal in degrees F but our reference temp was in degrees C")
]

for test_call, result, explanation in test_cases:
    print(f"{test_call}: {result}, #{explanation}")



# Question 2:
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime


# Question 2a:
data = pd.read_csv("https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv")
data['date'] = pd.to_datetime(data['date'])


# Question 2b:
# Calculate new cases
data = data.sort_values(['state', 'date'])
data['new_cases'] = data.groupby('state')['cases'].diff().fillna(0)

# Define the plotting function
def plot_states_new_cases(state_list):
    plt.figure(figsize=(12, 6))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    for i, state in enumerate(state_list):
        state_data = data[data['state'] == state].copy()
        
        if len(state_data) > 0:
            plt.plot(state_data['date'], state_data['new_cases'], 
                    color=colors[i % len(colors)], 
                    label=state, 
                    linewidth=2)
    
    plt.title('COVID-19 New Cases by State Over Time')
    plt.xlabel('Date')
    plt.ylabel('New Cases')
    plt.legend()
    
    # Format x-axis dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

# Test the function
state_list = ['New York', 'California', 'Texas', 'Florida']
plot_states_new_cases(state_list)


# Question 2c:
# Define the peak date function
def find_peak_date(state_name):
    state_data = data[data['state'] == state_name]
    
    peak_idx = state_data['new_cases'].idxmax()
    peak_date = state_data.loc[peak_idx, 'date']
    
    return peak_date

state = 'New York'  # Select the city of interest
peak_date = find_peak_date(state)  # run the function
peak_cases = data[(data['state'] == state) & (data['date'] == peak_date)]['new_cases'].iloc[0]  # Extract the information
print(f"{state}: {peak_date.strftime('%Y-%m-%d')} ({int(peak_cases):,} new cases)")  # Display the information


# Question 2d:
# Define the function
def compare_peak_cases(state1, state2):
    peak_date1 = find_peak_date(state1)
    peak_date2 = find_peak_date(state2)
    
    # Get peak case numbers
    peak_cases1 = data[(data['state'] == state1) & (data['date'] == peak_date1)]['new_cases'].iloc[0]
    peak_cases2 = data[(data['state'] == state2) & (data['date'] == peak_date2)]['new_cases'].iloc[0]
    
    # Determine which peaked first
    if peak_date1 < peak_date2:
        first_peak_state = state1
        days_between = (peak_date2 - peak_date1).days
    elif peak_date2 < peak_date1:
        first_peak_state = state2
        days_between = (peak_date1 - peak_date2).days
    else:
        first_peak_state = "Same day"
        days_between = 0
    
    return first_peak_state, days_between


# Test the function
first_state, days_diff = compare_peak_cases('New York', 'California')  # Change the city name with cities interested

print("New York vs California:")
print(f"{first_state} peaked first")
print(f"Days between peaks: {days_diff}")


# Question 2e:
plot_states_new_cases(['Florida'])
