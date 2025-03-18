from ManageData import PassTrain


print("please wait a moment...")
data = PassTrain("processed_data.csv")

print(data.countBy(**{"strength" : 0}))
print(data.countBy(**{"strength"  : 1}))
print(data.countBy(**{"strength" : 2}))



print(data.getBy(**{"total_len" : 4}))
print(data.countBy(**{ "total_len" : 4}))


print(data.getBy(**{"total_len" : 4, "upper_case_count" : 2}))
print(data.getBy(**{"total_len": 4, "upper_case_count" : 1, "lower_case_count" : 2}))
print(data.countBy(**{"total_len": 4, "upper_case_count" : 2}))
print(data.countBy(**{"total_len" : 4, "upper_case_count" : 1, "lower_case_count" : 2}))
