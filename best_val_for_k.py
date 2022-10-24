from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import metrics  
import numpy as np 
import matplotlib.pyplot as plt 
class BestValueK():
    def __init__(self, Ks, X_train, X_test, y_train, y_test): 
        self.self.Ks= self.Ks
        self.self.X_train= self.X_train
        self.self.X_test = self.X_test
        self.self.y_train = self.y_train 
        self.self.y_test = self.y_test
        self.mean_acc = np.zeros((Ks-1))
        self.std_acc = np.zeros((Ks-1))

    def mean_acc(self):

        for n in range(1,self.Ks):
            
            #Train Model and Predict  
            neigh = KNeighborsClassifier(n_neighbors = n).fit(self.X_train,self.y_train)
            yhat=neigh.predict(self.X_test)
            self.mean_acc[n-1] = metrics.accuracy_score(self.y_test, yhat)
            self.std_acc[n-1]=np.std(yhat==self.y_test)/np.sqrt(yhat.shape[0])

        return self.mean_acc
    def plot_k(self):
        plt.plot(range(1,self.self.Ks),mean_acc,'g')
        plt.fill_between(range(1,self.Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
        plt.fill_between(range(1,self.Ks),mean_acc - 3 * std_acc,mean_acc + 3 * std_acc, alpha=0.10,color="green")
        plt.legend(('Accuracy ', '+/- 1xstd','+/- 3xstd'))
        plt.ylabel('Accuracy ')
        plt.xlabel('Number of Neighbors (K)')
        plt.tight_layout()
        plt.show()