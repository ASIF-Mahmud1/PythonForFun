# Neural network implementation
from sklearn.neural_network import MLPClassifier
# Function for loading the breast cancer data set
from sklearn.datasets import load_breast_cancer
# Function for splitting data into training & testing sets
from sklearn.model_selection import train_test_split 
data = load_breast_cancer() # Load and save the breast cancer data set
attributes = data.data # The properties of breast mass cell nuceli 
labels = data.target # Whether the mass is cancerous
##############################
# Random Forest implementation
from sklearn.ensemble import RandomForestClassifier

# Preprocessing data

# Changed var name @neuralnetwork to @randomforest to avoid confusion
randomforest = RandomForestClassifier()


##########################
# print("Data is ",data)

attributes_train, attributes_test, labels_train, labels_test = train_test_split(attributes, labels, test_size=0.33)

# neuralnet = MLPClassifier() # Instantiate a neural network object
neuralnet = MLPClassifier(solver='lbfgs', activation='logistic', alpha=10.0) 
neuralnet.fit(attributes_train, labels_train) # Train the neural network
randomforest.fit(attributes_train, labels_train) # Train the neural network

accuracyForNeural = neuralnet.score(attributes_test, labels_test) # Test the neural network
accuracyForRandomForest= randomforest.score(attributes_test, labels_test)
print("Neural Network Accauracy ",str(accuracyForNeural * 100) + "% accuracy") # Print the accuracy

print("Random Forest Accauracy ",str(accuracyForRandomForest * 100) + "% accuracy") # Print the accuracy
