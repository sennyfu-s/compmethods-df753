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
    print(f"{test_call} # {result} -- {explanation}")
