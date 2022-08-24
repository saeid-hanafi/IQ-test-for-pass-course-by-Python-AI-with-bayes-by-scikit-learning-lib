# Libraries
# import library for read Excel files
import xlrd
# import library for convert python lists to matrix
import numpy as np
# import library for use AI By naive bayes
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics


# Functions
# function for read Excel files by location
def read_excel_files(file_loc):
    file = xlrd.open_workbook(file_loc)
    return file.sheet_by_index(0)


# read dateset Excel file by xls format
dataset_loc = "DataSet.xls"
read_dataset = read_excel_files(dataset_loc)
dataset = []
for data_index in range(0, read_dataset.nrows):
    dataset_items = [read_dataset.cell_value(data_index, 0), read_dataset.cell_value(data_index, 1)]
    dataset.append(dataset_items)

# create IQs list And Result list By dataset
x_train = []
y_train = []
for i in range(0, len(dataset)):
    x_train.append(dataset[i][0])
    y_train.append(dataset[i][1])

# convert IQs And Result lists to vertical matrix
IQs_list = np.array(x_train).reshape(-1, 1)
res_list = np.array(y_train).reshape(-1, 1)

# get date test from xls file the same above (TestSet.xls)
test_loc = "TestSet.xls"
read_test = read_excel_files(test_loc)
test_data = []
for j in range(0, read_test.nrows):
    test_items = [read_test.cell_value(j, 0), read_test.cell_value(j, 1)]
    test_data.append(test_items)

IQs_test = []
res_test = []
for k in range(0, len(test_data)):
    IQs_test.append(test_data[k][0])
    res_test.append(test_data[k][1])

IQs_test_list = np.array(IQs_test).reshape(-1, 1)
res_test_list = np.array(res_test).reshape(-1, 1)

# Final Results
# get final results by scikit learning library
gnb = GaussianNB()
gnb.fit(IQs_list, res_list)
result = gnb.predict(IQs_test_list)
print("List of IQs For Test : ", IQs_test)
print("Test Results : ", result)

# get accuracy score
avg_result_true = metrics.accuracy_score(res_test_list, result)
print(avg_result_true * 100, "% is True Answer")
