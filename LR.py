import csv

import numpy as np
import matplotlib.pyplot as plt

def split_data(X, Y, ratio):
    "Split the data based on the ratio."
    if X.size != Y.size:
        raise ValueError("Invalid vector shapes.")
    
    N = X.size
    A, B = X[:int(N * ratio)], Y[:int(N * ratio)]
    W, Z = X[int(N * ratio):], Y[int(N * ratio):]

    return A, B, W, Z

# Open the input file, write its content into two  arrays.
X = []
Y = []
with open("HW_data.csv") as f:
    reader = csv.reader(f)
    for row in reader:
        X.append(float(row[0]))
        Y.append(float(row[1]))

# Convert the arrays into numpy arrays and randomize their elements.
X = np.array(X)
Y = np.array(Y)

p = np.random.permutation(len(X))
X = X[p]
Y = Y[p]

# Split the data into train and test sets.
X_train, Y_train, X_test, Y_test = split_data(X, Y, 0.8)

class LR:
    def __init__(self, epochs, lr):
        self.lr = lr
        self.epochs = epochs
        self.slope = 0
        self.intercept = 0

    def cost(self, X, Y):
        "Calculate the cost using MSE function."
        return np.sum((Y - self.predict(X)) ** 2) / len(X)

    def train(self, X, Y, train_validation_ratio, show_learning_cost=False):
        """
            Split the data into test and validation test.
            Then run the gradient descent on the train data and 
            use the validation data to make sure that the cost is decreasing.
            If the `show_learning_cost` is True, plot the learning cost 
            for train and validation sets from the beginning.
        """
        X_train, Y_train, X_validate, Y_validate = split_data(X, Y, train_validation_ratio)

        X_costs = []
        Y_costs_train = []
        Y_costs_validate = []

        for i in range(self.epochs):
            dslope = -2 * np.sum((Y_train - self.predict(X_train)) * X_train) / len(X_train)
            dintercept = -2 * np.sum(Y_train - self.predict(X_train)) / len(X_train)

            self.slope -= self.lr * dslope
            self.intercept -= self.lr * dintercept

            if show_learning_cost:
                X_costs.append(i)
                Y_costs_train.append(np.sum(self.cost(X_train, Y_train)))
                Y_costs_validate.append(np.sum(self.cost(X_validate, Y_validate)))

                if len(Y_costs_validate) > 2 and Y_costs_validate[i] > Y_costs_validate[i - 1]:
                    break
        
        if show_learning_cost:
            f1 = plt.figure(1)
            plt.title("Sum of Cost Functions for Train and Validation Datasets")
            plt.xlabel("x")
            plt.ylabel("Sum of cost functions")
            plt.plot(X_costs, Y_costs_train, label="Train costs")
            plt.plot(X_costs, Y_costs_validate, label="Validate costs")
            plt.legend()
            f1.show()

    def predict(self, X):
        return self.slope * X + self.intercept

    def params(self):
        return self.slope, self.intercept

model = LR(1000, 0.01)
model.train(X_train, Y_train, 0.8, show_learning_cost=True)
coefficient, intercept = model.params()

f2 = plt.figure(2)
plt.scatter(X, Y, s=20, label="Data samples")
plt.plot([-3, 3], [model.predict(-3), model.predict(3)], color="r", linewidth=3, label="Best fitted line")
plt.title("Linear Regression")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
f2.show()

print("Weights: ", model.params())
print("Costs on Test dataset: ", np.sum(model.cost(X_test, Y_test)))

input("Press enter to continue...")
