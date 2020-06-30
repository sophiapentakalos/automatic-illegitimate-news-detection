from sklearn import svm

# Read data
fake_data = open("fakeData2ft.txt", 'r')
legit_data = open("legitData2ft.txt", 'r')
fake_data = fake_data.read().split('\n')
legit_data = legit_data.read().split('\n')
total_fakes =  len(fake_data) - 1
total_legits = len(legit_data) - 1
training_ind = int(.8 * total_fakes)
#creating training data set
training_x = []
for j in range(training_ind):
    line_arr = fake_data[j].split()
    line_arr = [float(i) for i in line_arr]
    training_x.append(line_arr)
training_labels_ind1 = len(training_x)
for j in range(training_ind):
    line_arr = legit_data[j].split()
    line_arr = [float(i) for i in line_arr]
    training_x.append(line_arr)
training_labels_ind2 = len(training_x) - training_labels_ind1
training_labels = [0] * training_labels_ind1 + [1] * training_labels_ind2

#creating testing data set
testing_x = []
for j in range(training_ind, total_fakes):
    line_arr = fake_data[j].split()
    line_arr = [float(i) for i in line_arr]
    testing_x.append(line_arr)
testing_labels_ind1 = len(testing_x)
for j in range(training_ind, total_legits):
    line_arr = legit_data[j].split()
    line_arr = [float(i) for i in line_arr]
    testing_x.append(line_arr)
testing_labels_ind2 = len(testing_x) - testing_labels_ind1
testing_labels = [0] * testing_labels_ind1 + [1] * testing_labels_ind2

#train classifier
clf = svm.SVC()
clf.fit(training_x, training_labels)

#test classifier
prediction = clf.predict(testing_x)
TP = 0
TN = 0
FP = 0
FN = 0
for i in range(len(prediction)):
    if prediction[i] == 1:
        if testing_labels[i] == 1:
            TP += 1
        else:
            FP += 1
    elif testing_labels[i] == 0:
        TN += 1
    else:
        FN += 1

precision = TP/(TP + FP)
recall = TP/(TP + FN)