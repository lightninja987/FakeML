#Imports necessary modules for the program to run
import pandas
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix
from matrix import plot_confusion_matrix
import time
from sklearn.metrics import classification_report
#Begins timer for program
initTime = time.time()

#Reads the Comma Separated Value file into the dataframe
file = pandas.read_csv(r"C:\Users\rohan\Downloads\fake_or_real_news.csv")

"""
Sets y to the labels(FAKE or REAL) of the file and also drops the 
first row of the dataset since it is not necessary
"""
y = file.label
file.drop("label",axis = 1)

"""
Chooses random files in dataset to be training and testing data. test_size shows what portion of the data
will be test data. random_state is used for generating random numbers to help
"""
X_train, X_test, y_train, y_test = train_test_split(file['text'], y, test_size=0.4,random_state=53)

       
#Stores tokens as numerical indexes

hash_vect = HashingVectorizer(stop_words='english', non_negative=True)


#fits the data to make a normal model
hash_train = hash_vect.fit_transform(X_train)
hash_test = hash_vect.transform(X_test)



#Creates instance of passive aggressive classifier
classifier = PassiveAggressiveClassifier()
#fit classifier onto training data
classifier.fit(hash_train,y_train)
#using 'learned' features from training data, predicts whether news is fake or real
prediction = classifier.predict(hash_test)




accuracy = accuracy_score(y_test, prediction) * 100
#print out total accuracy of classifier
print("The accuracy is %0.5f" % accuracy + " percent.")

#creates confusion matrix
matrix = confusion_matrix(y_test, prediction, labels=['FAKE', 'REAL'])
plot_confusion_matrix(matrix, classes=['FAKE', 'REAL'])
#prints out classification report
print(classification_report(y_test, prediction))
#prints out most predictive features for classification
print("The program ran in " + str(round(time.time() - initTime,3)) + " seconds.")
