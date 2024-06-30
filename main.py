from ManageData import PassTrain

password = input("give a password to score\n~>>")
print("processing please wait")
data = PassTrain("processed_data.csv")

score = data.score_password(password)


if score == 0:
    print("your password is pretty bad, here is how you can improve it:")
    print(data.suggestImprovements(password))
if score == 1:
    print("your password is good, but can do better:")
    print(data.suggestImprovements(password))
if score == 2:
    print(data.suggestImprovements(password))


print(f"Overall time: {PassTrain.overall_time:.2f} seconds")
