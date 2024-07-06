from ManageData import PassTrain

data = PassTrain("processed_data.csv", input("give a password to score\n~>>"))
print("processing please wait")

# note if you wanted to use it how it was before when you do data.score_passowrd(<input>, do_guess = True)


if data.score == 0: 
    print("your password is pretty bad, here is how you can improve it:")
    print(data.suggestImprovements(data.improve))
if data.score == 1:
    print("your password is good, but can do better:")
    print(data.suggestImprovements(data.improve))
if data.score == 2:
    print(data.suggestImprovements(data.improve))

print(f"Overall time: {PassTrain.overall_time:.2f} seconds")
