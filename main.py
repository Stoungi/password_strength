from ManageData import PassTrain


print("please wait a moment...")
data = PassTrain("processed_data.csv")



while True:

    choice = int(input("what would you like to do?\n 1. rate password\n 2. history\n 3. exit\n~>>"))

    if choice == 3:
        break
    elif choice == 2:
        for password, improve in data.history:
            print(f"{password}\n{improve}")
    elif choice == 1:
        data.score_password(input("give a password to score\n~>>"))
        if data.score == 0:
            print("your password is pretty bad, here is how you can improve it:")
            print(data.improve)
        if data.score == 1:
            print("your password is good, but can do better:")
            print(data.improve)
        if data.score == 2:
            print(data.improve)


    else:
        pass
