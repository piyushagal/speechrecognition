__author__ = 'piyush'
import sys
from Tkinter import *
from Record import Recorder
from learner import *
import cPickle

class Window:
    def startWindow(self, l):
        self.root = Tk()
        self.root.geometry('350x100')
        self.learn = l
        frame1 = Frame(self.root)
        self.l = Label(frame1, text='Hello World !!')
        b = Button(frame1, text = 'Record', command = self.recordNdetect)
        frame2 = Frame(self.root)
        for method in ["1","2","3","4","5","6","7","8","9","10"]:
            button = Button(frame2, text=method,
                        command=lambda m=method: self.onlineLearn(m))
            button.pack({'side': 'left'})

        self.l.pack(side=TOP)
        b.pack(side=BOTTOM)
        frame1.pack()
        frame2.pack()
        self.root.mainloop()

    def stopWindow(self):
        self.root.destroy()

    def setLabel(self, value):
        self.l.configure(text = value)

    def recordNdetect(self):
        r = Recorder()
        data = r.record()
        r.save(data,"./TestData1/output.wav")
        self.speech = Speech('./TestData1/', 'output.wav')
        self.speech.extractFeature()
        self.setLabel(self.recognize())

    def onlineLearn(self, id):
        self.speech.categoryId = int(id)-1
        self.learn.speechRecognizerList[int(id)-1].trainData.append(self.speech.features)
        self.learn.speechRecognizerList[int(id)-1].hmmModel.fit(self.learn.speechRecognizerList[int(id)-1].trainData)
        with open('hmmModel.pkl','wb') as fid:
            cPickle.dump(self.learn, fid)
        print('Done Training')


    def recognize(self):
        scores = []
        for recognizer in self.learn.speechRecognizerList:
            score = recognizer.hmmModel.score(self.speech.features)
            scores.append(score)

        idx1 = scores.index(max(scores))
        scores[idx1] = -10000
        idx2 = scores.index(max(scores))
        predictCategoryId = self.learn.speechRecognizerList[idx1].categoryId + ',' + self.learn.speechRecognizerList[idx2].categoryId

        return predictCategoryId

print(__name__)
if __name__  ==  '__main__':
    print('started')
    if os.path.exists('hmmModel.pkl'):
        with open('hmmModel.pkl','r') as fid:
             l = cPickle.load(fid)
    else:
        l = learn()
   # l = learn()
    w = Window()
    w.startWindow(l)




