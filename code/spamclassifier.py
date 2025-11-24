import numpy as np
import pandas as pd
import string
import matplotlib.pyplot as pyplot

import nltk
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve, precision_recall_curve

nltk.download("stopwords")

#Using Naive Bayes to classify emails as spam or ham (not spam)
#Based off of the following paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10486769

#Preprocessing function (from the paper)
#This is the only code I took from the paper, they didn't have anything else
def processText(unprocessed):
    #Remove punctuation
    nopunc = [char for char in unprocessed if char not in string.punctuation]
    nopunc = ''.join(nopunc)

    #Remove stop words
    processed = [word for word in nopunc.split() if word.lower() not in stopwords.words("english")]
    return ' '.join(processed)

#Function to import the dataset (created for resuse in milestone 3)
def importDataset():
    #Load the dataset and change column names
    print("\nLoading the dataset...")
    df = pd.read_csv("../spam.csv", encoding="latin-1", skiprows=1, usecols=[0, 1], names=["label", "text"])

    #Process the data and put it into a new column
    print("Processing the data...\n")
    df["clean text"] = df["text"].apply(processText)

    #Test prints for preprocessing
    '''print(df[["label", "clean text"]].head())
    print(df["clean text"].iloc[0])'''

    return df

def evaluate(labels, preds, probs, modelType):
    #Get the accuracy and classification report
    accuracy = accuracy_score(labels, preds)
    print(f"Accuracy: {accuracy*100:.2f}%")
    report = classification_report(labels, preds, target_names=["ham", "spam"])
    print(report)

    #Display the confusion matrix
    confusionMatrix = confusion_matrix(labels, preds)
    matrix = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix, display_labels=["ham", "spam"])
    matrix.plot()

    #Compute AUROC
    auroc = roc_auc_score(labels, probs)

    #Plot the ROC curve
    if modelType == "NB": pos_label = "spam"
    else: pos_label = 1
    fp_rate, tp_rate, _ = roc_curve(labels, probs, pos_label=pos_label)

    pyplot.figure()
    pyplot.plot(fp_rate, tp_rate, label=f'ROC curve (area = {auroc:.4f})')
    pyplot.xlabel('fp-rate')
    pyplot.ylabel('tp-rate')
    pyplot.title('ROC Curve')
    pyplot.legend(loc='lower right')
    pyplot.show()

    #Plot the precision-recall curve
    precision, recall, _ = precision_recall_curve(labels, probs, pos_label=pos_label)

    pyplot.figure()
    pyplot.plot(recall, precision)
    pyplot.xlabel('Recall')
    pyplot.ylabel('Precision')
    pyplot.title('Precision-Recall Curve')
    pyplot.show()

if __name__ == "__main__":
    seed = int(input("\nEnter seed for split randomization (integer input | recommended=4): "))

    df = importDataset()

    #Next step is to convert the messages into vectors
    #The paper uses TF-IDF for this
    vectorizer = TfidfVectorizer()
    #Apply TF-IDF to the processed messages, creating a messages-by-words matrix
    vectorized = vectorizer.fit_transform(df["clean text"])

    #Then the data is split into train/val (80%) and test (20%) sets
    #The random_state gives a seed for randomizing the data across splits
    #Stratify ensures that each split contains approximately the same (proportional) amount of each class
    trainText, testText, trainLabels, testLabels = train_test_split(vectorized, df["label"], test_size=0.2, random_state=seed, stratify=df["label"])

    #Now we train the model
    print("Training the model...")
    #Generate an array of alpha values between 0.1 and 2 with step size 0.1 for testing
    alphas = np.arange(0.1, 2.1, 0.1)
    bestAlpha = None
    bestAccuracy = 0
    bestStd = 0

    #Peform K-fold cross validation as requested for the assignment
    K = 10
    kfold = StratifiedKFold(n_splits=K, shuffle=True, random_state=seed)

    #We can use the opportunity presented by the cross validation to tune the alpha hyperparameter in multinomial NB
    for a in alphas:
        NB = MultinomialNB(alpha=a)
        accuracies = cross_val_score(NB, trainText, trainLabels, cv=kfold, scoring='accuracy')
        meanAccuracy = np.mean(accuracies)
        #print(f"Alpha = {a:.1f} - Cross-val mean accuracy: {meanAccuracy*100:.2f}%")

        if meanAccuracy > bestAccuracy:
            bestAccuracy = meanAccuracy
            bestStd = np.std(accuracies, ddof=1)
            bestAlpha = a

    print(f"\nCross-val mean accuracy: {bestAccuracy*100:.2f}% | Standard deviation: {bestStd:.5f}")
    print(f"Best alpha: {bestAlpha:.1f}")

    #Perform the t-test for H0: p >= p0 vs H1: p < p0 (left-tailed t-test)
    #Where p is my model accuracy and p0 is the paper's model accuracy
    p0 = 0.96
    tValue = (np.sqrt(K) * (bestAccuracy - p0)) / bestStd
    criticalValue = 1.833       #From t-table: 1 sided with alpha = 0.05
    print(f"\nt value: {tValue} | critical value: {criticalValue}")

    if tValue < criticalValue:
        print("Reject H0 at significance level 0.05: My model's accuracy is significantly worse than the paper's (H0: p >= p0)")
    else:
        print("Fail to reject H0 at significance level 0.05: My model's accuracy is at least as good as the paper's (H1: p < p0)")


    #Now test the accuracy of the model on the test set using the best alpha
    testNB = MultinomialNB(alpha=bestAlpha)
    testNB.fit(trainText, trainLabels)

    print("\nTesting the model...")
    #Make predictions on the test data
    testPredictions = testNB.predict(testText)

    #Get the predicted probability that each message is spam
    testProbs = testNB.predict_proba(testText)[:, 1]

    print("Generating test results...\n")
    evaluate(testLabels, testPredictions, testProbs, "NB")