import os

from utils import Speech, SpeechRecognizer
from sklearn import linear_model, svm, metrics
from scipy.ndimage import convolve
from sklearn.pipeline import Pipeline
from sklearn.neural_network import BernoulliRBM
import numpy as np
from sklearn import svm, tree, lda, neighbors, naive_bayes, cross_validation, linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import VarianceThreshold

CATEGORY = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']

class Learning:

    def featureReduction(self,X):
        sel = VarianceThreshold(threshold=(0.01*(1-0.01)))
        X_new = sel.fit_transform(X)
        if len(X_new[0])!=len(X[0]):
            print('Feature Vector Dimension Reduced by ' + str(len(X[0]) - len(X_new[0])))
        return X_new

    def loadData(self,dirName):
        fileList = [f for f in os.listdir(dirName) if os.path.splitext(f)[1] == '.wav']

        speechList = []

        for fileName in fileList:
            speech = Speech(dirName, fileName)
            speech.extractFeature()
            speechList.append(speech)

        return speechList

    def training(self,speechList):

        speechRecognizerList = []

        for categoryId in CATEGORY:
            speechRecognizer = SpeechRecognizer(categoryId)
            speechRecognizerList.append(speechRecognizer)

        for speechRecognizer in speechRecognizerList:
            for speech in speechList:
                if(speech.categoryId == speechRecognizer.categoryId):
                    speechRecognizer.trainData.append(speech.features)

            speechRecognizer.initModelParameter(nComp=5, nMix = 2,  covarianceType='diag', n_iter=20, bakisLevel=2)
            speechRecognizer.getHmmModel()

        self.speechRecognizerList = speechRecognizerList

    def trainingNeuralNetwork(self,speechList):

        X = []
        Y = []
        for speech in speechList:
            x = speech.features.flatten()
            a = 12000 - len(x)
            x = np.lib.pad(x,(0,a),mode = 'constant', constant_values=0)
            X.append(x)
            Y.append(speech.categoryId)

        logistic = self.getNeuralModel(X,Y)

        return logistic

    def getNeuralModel(self,X,Y):

            logistic = linear_model.LogisticRegression()
            rbm = BernoulliRBM(verbose=True)

            classifier = linear_model.LogisticRegression(penalty='l2', tol=.0001)#Pipeline(steps = [('rbm', rbm),('logistic',logistic)])
            rbm.learning_rate = 0.0001
            rbm.n_iter = 1000
            rbm.n_components = 1000

            classifier.fit(X, Y)

            return classifier


    def recognize(self,testSpeechList):
        ''' recognition '''
        predictCategoryIdList = []

        for testSpeech in testSpeechList:
            scores = []

            for recognizer in self.speechRecognizerList:
                score = recognizer.hmmModel.score(testSpeech.features)
                scores.append(score)

            idx = scores.index(max(scores))
            predictCategoryId = self.speechRecognizerList[idx].categoryId
            predictCategoryIdList.append(predictCategoryId)

        return predictCategoryIdList

    def recognizeNeural(self,testSpeechList, logistic):
        ''' recognition '''
        predictCategoryIdList = []

        X = []
        Y = []
        for testSpeech in testSpeechList:
             x = testSpeech.features.flatten()
             a = 12000 - len(x)
             x = np.lib.pad(x,(0,a),mode = 'constant', constant_values=(0))
             predictCategoryIdList.append(logistic.predict(x)[0])

        return predictCategoryIdList


    def calculateRecognitionRate(self,groundTruthCategoryIdList, predictCategoryIdList):
        ''' calculate recognition rate '''
        score = 0
        length = len(groundTruthCategoryIdList)

        for i in range(length):
            gt = groundTruthCategoryIdList[i]
            pr = predictCategoryIdList[i]

            if gt == pr:
                score += 1

        recognitionRate = float(score) / length
        return recognitionRate

def learn():
    l = Learning()
    print("1. Loading training data...")
    trainDir = "./eng_training_data/"
    trainSpeechList = l.loadData(trainDir)
    print("done!")

    print("2. Training Model...")
    speechRecognizerList = l.training(trainSpeechList)
    logistic = l.trainingNeuralNetwork(trainSpeechList)
    print("done!")

    print("3. Test Data Loading...")
    testDir = "./eng_test_data/"
    testSpeechList = l.loadData(testDir)
    print("done!")

    print("4. Recognizing...")
    predictCategoryList = l.recognize(testSpeechList)
    predictCategoryListNeural = l.recognizeNeural(testSpeechList, logistic)
    print("done!")


    groundTruthCategoryIdList = [speech.categoryId for speech in testSpeechList]
    recognitionRate = l.calculateRecognitionRate(groundTruthCategoryIdList, predictCategoryList)

    print '===== Final result ====='
    print 'Ground Truth:\t', groundTruthCategoryIdList
    print 'Prediction:\t', predictCategoryList
    print 'Accuracy:\t', recognitionRate

    recognitionRate = l.calculateRecognitionRate(groundTruthCategoryIdList, predictCategoryListNeural)

    print '===== Final result ====='
    print 'Ground Truth:\t', groundTruthCategoryIdList
    print 'Prediction:\t', predictCategoryListNeural
    print 'Accuracy:\t', recognitionRate

    return l


