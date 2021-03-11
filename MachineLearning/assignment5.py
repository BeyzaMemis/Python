import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt  # 2.2.4
import pandas as pd  # 0.24.2
import numpy as np  # 1.16.4

# python 2.7

csv_f = "breast-cancer-wisconsin.csv"

# reading  csv file
df = pd.read_csv(csv_f)

# making new df without missing values
new_df = df.loc[~(df['Bare_Nuclei'] == '?')]
new_df = new_df.reset_index()

# making training and test csv
number_of_fd = len(new_df.index)
number_of_trains = int(0.8 * number_of_fd)
new_df.head(number_of_trains).to_csv("modified_training.csv", index=False)
new_df.tail(number_of_fd - number_of_trains).to_csv("modified_test.csv", index=False)


# pairwise visulation kod
dt = pd.read_csv('modified_training.csv')
dt = dt.drop(columns=['Code_number'])
dt = dt.drop(columns=['Class'])
dt = dt.drop(columns=['index'])

correlations = dt.corr().round(2)
names = ['Clump_Thickness', 'Uniformity_of_Cell_Size', 'Uniformity_of_Cell_Shape', 'Marginal_Adhesion',
         'Single_Epithelial_Cell_Size', 'Bare_Nuclei', 'Bland_Chromatin', 'Normal_Nucleoli', 'Mitoses']

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=0, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0, 9, 1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.setp(ax.get_xticklabels(), rotation=45, ha="left",
         rotation_mode="anchor")

for i in range(len(correlations.columns)):
    for j in range(len(correlations.columns)):
        text = ax.text(j, i, (correlations.iloc[i, j]),
                       ha="center", va="center", color="black")

plt.show()


#

def sigmoid(x):
    return 1 / (1 + np.exp(-0.005 * x))


def sigmoid_derivative(x):
    return 0.005 * x * (1 - x)


def readcsv(filename):
    data = pd.read_csv(filename)
    data = data.drop(columns=['Code_number'])
    data = data.drop(columns=['index'])
    df_class = data.iloc[:, 9]
    data = data.drop(columns=['Class'])

    return np.array(data), np.array(df_class)


def run_on_test_set(test_inputs, test_labels, weights):
    tp = 0

    test_outputs = []
    # calculate test_predictions
    # TODO map each prediction into either 0 or 1
    for k in range(len(test_inputs)):
        test = test_inputs[k, :]
        test_output = sigmoid(np.dot(test, weights))
        test_outputs.append(test_output)
    test_predictions = (map(lambda a: 1 if a > 0.5 else 0, test_outputs))

    for predicted_val, label in zip(test_predictions, test_labels):
        if predicted_val == label:
            tp += 1
    length = int(len(test_labels))
    accur = float(tp) / float(length)
    return accur


training_inputs, trainig_labels = readcsv("modified_training.csv")
trainig_labels = np.resize(trainig_labels, (len(trainig_labels), 1))

test_inputs, test_labels = readcsv("modified_test.csv")

weights = 2 * np.random.random((9, 1)) - 1

accuracy = []
loss_ = []

for i in range(2500):
    input = training_inputs
    labels = trainig_labels

    # forward
    out_puts = np.dot(input, weights)

    out_puts = sigmoid(out_puts)
    # backward
    loss = labels - out_puts

    loss_mean = np.mean(loss)
    loss_.append(loss_mean)

    tuning = loss * sigmoid_derivative(out_puts)

    trans = training_inputs.transpose()

    weights += np.dot(trans, tuning)

    accuracy.append(run_on_test_set(test_inputs, test_labels, weights))


def plotting(ary, name):
    x = np.arange(0, len(ary))
    y = ary

    plt.plot(x, y)
    plt.title(name + ' Graph')
    plt.xlabel('Iteration counts')
    plt.ylabel(name + 'Values')
    plt.show()


# plotting accuracy

plotting(accuracy, 'Accuracy')

# plotting loss

plotting(loss_, 'loss')
