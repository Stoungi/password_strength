from ManageData import PassTrain


print("please wait a moment...")
data = PassTrain("processed_data.csv")

print(data.countBy(**{data._strength : 0}))
print(data.countBy(**{data._strength : 1}))
print(data.countBy(**{data._strength : 2}))



print(data.getBy(**{data._total_len : 4}))
print(data.countBy(**{data._total_len : 4}))


print(data.getBy(**{data._total_len : 4, data._upper_case_count : 2}))
print(data.getBy(**{data._total_len : 4, data._upper_case_count : 1, data._lower_case_count : 2}))
print(data.countBy(**{data._total_len : 4, data._upper_case_count : 2}))
print(data.countBy(**{data._total_len : 4, data._upper_case_count : 1, data._lower_case_count : 2}))