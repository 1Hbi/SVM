# Import necessary libraries
import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

# Load the Breast Cancer dataset
cancer = datasets.load_breast_cancer()

# Uncomment the following lines to print feature names and target names
# print(cancer.feature_names)
# print(cancer.target_names)

# Separate the features (x) and target variable (y) from the dataset
x = cancer.data
y = cancer.target

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

# Print the training data and labels
print("Training Data:", x_train)
print("Training Labels:", y_train)

# Define class labels for better interpretation of predictions
classes = ['malignant', 'benign']

# Initialize a Support Vector Classifier with a linear kernel
clf = svm.SVC(kernel='linear', C=1.0)
# Alternatively, use K-Nearest Neighbors classifier (uncomment the following line)
# clf = KNeighborsClassifier(n_neighbors=7)

# Train the classifier on the training data
clf.fit(x_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(x_test)

# Calculate and print the accuracy of the classifier
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
