from scikits.talkbox.features import mfcc
from scipy.io import wavfile
from hmmlearn import hmm
from sklearn import linear_model, datasets, metrics
from sklearn.pipeline import Pipeline
from sklearn.neural_network import BernoulliRBM
import numpy as np

class Speech:
    ''' speech class '''

    def __init__(self, dirName, fileName):
        self.fileName = fileName    # file name
        self.features = None    # feature matrix
        self.soundSamplerate, self.sound = wavfile.read(dirName + fileName)

        # category assignment
        idx1 = self.fileName.find('_')
        idx2 = self.fileName.find('.')
        self.categoryId = fileName[idx1 + 1 : idx2]   # speech category


    def extractFeature(self):
        ''' mfcc feature extraction '''

        self.features = mfcc(self.sound, nwin=int(self.soundSamplerate * 0.03), fs=self.soundSamplerate, nceps=13)[0]

class SpeechRecognizer:
    ''' class for speech recognizer '''

    def __init__(self, categoryId):
        self.categoryId = categoryId
        self.trainData = []
        self.hmmModel = None

        self.nComp = 5  # number of states
        self.nMix = 2   # number of mixtures
        self.covarianceType = 'diag'    # covariance type
        self.n_iter = 10    # number of iterations
        self.startprobPrior = None
        self.transmatPrior = None
        self.bakisLevel = 2

    def initModelParameter(self, nComp, nMix, covarianceType, n_iter, bakisLevel):
        ''' init params for hmm model '''

        self.nComp = nComp  # number of states
        self.nMix = nMix   # number of mixtures
        self.covarianceType = covarianceType    # covariance type
        self.n_iter = n_iter    # number of iterations
        self.bakisLevel = bakisLevel

        startprobPrior, transmatPrior = self.initByBakis(nComp, bakisLevel)
        self.startprobPrior = startprobPrior
        self.transmatPrior = transmatPrior

    def initByBakis(self, nComp, bakisLevel):
        ''' init start_prob and transmat_prob by Bakis model '''
        startprobPrior = np.zeros(nComp)
        startprobPrior[0 : bakisLevel - 1] = 1./ (bakisLevel - 1)

        transmatPrior = self.getTransmatPrior(nComp, bakisLevel)

        return startprobPrior, transmatPrior

    def getTransmatPrior(self, nComp, bakisLevel):
        ''' get transmat prior '''
        transmatPrior = (1. / bakisLevel) * np.eye(nComp)

        for i in range(nComp - (bakisLevel - 1)):
            for j in range(bakisLevel - 1):
                transmatPrior[i, i + j + 1] = 1. /  bakisLevel

        for i in range(nComp - bakisLevel + 1, nComp):
            for j in range(nComp - i -j):
                transmatPrior[i, i + j] = 1. / (nComp - i)

        return transmatPrior

    def getHmmModel(self):
        ''' get hmm model from training data '''

        # GaussianHMM
        #         model = hmm.GaussianHMM(numStates, "diag") # initialize hmm model

        # Gaussian Mixture HMM

        model = hmm.GaussianHMM(n_components=self.nComp,startprob_prior=self.startprobPrior, transmat_prior=self.transmatPrior, covariance_type='diag')
        model.fit(self.trainData)   # get optimal parameters

        self.hmmModel = model
