#Logistic Regression Mini 1

# Libraries import
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
# %matplotlib inline
# %matplotlib notedebook
import matplotlib.pyplot as plt


from ucimlrepo import fetch_ucirepo

# fetch dataset
wine = fetch_ucirepo(id=109)


# data (as pandas dataframes)
X = wine.data.features
Y = wine.data.targets

#Change them to array with .values
X = X.to_numpy()
Y = Y.to_numpy().flatten("C") #Flattening the array since it was an array of nested single value arrays, "C" collapses on certain column

X = (X - np.mean(X,axis = 0)) / np.std(X, axis = 0)

#This is to calculate the softmax value from the array of x values
def softmax(x):
    top = np.exp(x - np.max(x, axis = 1, keepdims = True))
    return top/np.sum(top, axis = 1, keepdims=True)
    

#This returns a an array of gradients associated to each feature with unique values for each class
def softmax_gradient(y_true, x, W):
    
    #First we compute the softmax probabilities
    y_pred = softmax(np.dot(x,W))

    #Set the weight gradient to an array with the same shape as the weight array
    gradient = np.zeros(np.shape(W), dtype = float) #Must sepcify type or there will be an error (why I don't know)

    #So we calculate the weight gradient for each feature and each class ( W is a F * C )
    for features in range(len(W)):
        for samples in range(len(x)):
            gradient[features] += x[samples][features] * (y_pred[samples] - y_true[samples])
        gradient[features] *= (1/len(x))

    return gradient


class Softmax_Logistic_Regression():

    def __init__(self, learning_rate, epsilon, max_iterations, num_of_classes, reg, minibatches, SGD = False, test = False):
        
        self.num_classes = num_of_classes

        self.learning_rate = learning_rate
        
        self.epsilon = epsilon

        self.max_iter = max_iterations

        self.reg = reg

        self.minibatches = minibatches

        self.SGD = SGD

        self.test = test #When testing, will print % finished in fitting for every 10% of iterations are finished (just in case you stall for whatever reason)

    def Y_reclassify(self, Y):

        #If Y is not currently one-hot encoded for softmax use, return an array that is
        if (np.shape(Y) != (len(Y), self.num_classes)):
            new_Y = np.zeros((len(Y), self.num_classes))
            for i in range(len(Y)):
                new_Y[i][int(Y[i] - 1)] = 1
            return new_Y
        
        else: #If Y is already in that form, do nothing
            return Y

    def fit(self, Y, X):

        Y = self.Y_reclassify(Y) #Reclassify Y values if necessary

        self.W = np.zeros((len(X[0]), self.num_classes)) #Make an array with # variables for rows and # of classes for columns
        
        g = np.inf
        
        t = 0

        #Setting up progress tracker (not necessary)
        if (self.test):
            checkpoint = int(self.max_iter*0.1)
            percent = 0

        while (t != self.max_iter) :

            if (self.SGD):
                n_test = int(len(Y)*(1/self.minibatches))
                inds = np.random.permutation(len(Y))
                for i in range(n_test):
                    Y_Train, X_Train = Y[inds[i*n_test:n_test*(i+1)]], X[inds[i*n_test:n_test*(i+1)]]

                    #If for any reason we need to skip (small minibatch due to rounding most likely)
                    if (len(X_Train) == 0):
                        continue

                    g = softmax_gradient(Y_Train, X_Train, self.W)
                    self.W = self.W - (g + (np.abs(self.W) * self.reg)) * self.learning_rate
                    if (np.max(np.sum(np.abs(g),axis =0)) <= self.epsilon):
                        return t


            else:
                g  = softmax_gradient(Y, X, self.W)

                #we update weights using W = W - (gradient + weight * regularization constant) * learning rate
                self.W = self.W - (g + (np.abs(self.W) * self.reg)) * self.learning_rate
                
                if (np.max(np.sum(np.abs(g),axis =0)) <= self.epsilon):
                        return t

            t += 1
                #Tells us how close we are to finished (can be turned off by not constructing the model with the True term at the end)
            if (self.test and t%checkpoint == 0):
                print(percent,"%")
                percent += 10

        return t
    #Given a certain set of test data, return the expected class of each sample
    def predict(self, X_given):

        probs = softmax(np.dot(X_given, self.W))

        prediction = np.zeros(len(X_given))

        for i in range(len(X_given)):
            max = 0
            for j in range(3):
                if (probs[i][j] > probs[i][max]):
                    max = j

            prediction[i] = max + 1 #We add 1 since index i is for class i + 1
        
        return prediction
    
    def get_W(self):
        return self.W
    
    def accuracy(self, x_test, y_true): 
        x2 = self.predict(x_test) #Predict for certain x values
        sum = 0
        for i in range(len(x2)):
            if (x2[i] == y_true[i]):
                sum += 1
        return sum/len(x_test)


    #We could always put all these functions in the class as methods (no need just a comment)
    def F1_score_with_P_and_R(self, y_pred, y_true):
        #Set up vectors for True Positives (TP), False Positives (FP) and False Negatives (FN)
        TP_vector = np.zeros(3)
        FP_vector = np.zeros(3)
        FN_vector = np.zeros(3)

        for i in range(len(y_true)):

            current = y_true[i]
        
            prediction = y_pred[i]

            #If guess is wrong
            if (prediction != current):

                FN_vector[int(current - 1)] += 1 #Increment false negative for class that it should have been part of

                FP_vector[int(prediction - 1)] += 1 #Increment false positive for class we predicted it belongs to
        
        #If guess if right
            else:
                TP_vector[int(current - 1)] += 1
    
        #Precision = TP/(TP + FP)
        precision = TP_vector/(TP_vector + FP_vector)

        #Recall = TP/(TP + FN)
        recall = TP_vector/(TP_vector + FN_vector)

        F1 = (2*precision*recall)/(precision + recall)

        #We are using arrays the entire time so the returned array has the F1 score for class = i at index (i - 1)
        return F1, recall, precision

def Exp1():
    model = Softmax_Logistic_Regression(0.1, 0.1, 100, 3, 0.1, 1 , False, False)
    y = Y
    x = X

    #Code shown in Collab notes for generalization to split dataset, modified for our purposes
    n_test = int(len(y)*0.2) #Isolate 20% of dataset for testing

    inds = np.random.permutation(len(y))

    Y_Test, X_Test = y[inds[:n_test]], x[inds[:n_test]]

    Y_Train, X_Train = y[inds[n_test:]], x[inds[n_test:]]

    #Fit the model with training data
    model.fit(Y_Train, X_Train)
    F1, Precision, Recall = model.F1_score_with_P_and_R( y_pred = (model.predict(X_Train)),y_true = Y_Train)
    print("Training Set: ", "\nF1_score: ", F1, "\nPrecison: ", Precision, "\nRecall: ", Recall, "\nAccuracy: ", model.accuracy(X_Train,Y_Train))

    F1, Precision, Recall = model.F1_score_with_P_and_R(y_pred = (model.predict(X_Test)),y_true = Y_Test)
    print("Testing Set: ", "\nF1_score: ", F1, "\nPrecison:", Precision, "\nRecall: ", Recall, "\nAccuracy: ", model.accuracy(X_Test,Y_Test))

    print("\nCombined Accuracy", model.accuracy(x,y))

def Exp2():
    model = Softmax_Logistic_Regression(0.1,0.1,100,3,0.1,1)
    x = X
    y = Y
    #Now we spilt the data into test and train (Using sciikit as intructed in assignment document)
    KFoldExp2 = KFold(n_splits=5,shuffle=True,random_state=10) #Using random_state = 10 for reproducable shuffling, or consistency
    model_num  = 1
    Score_Holder = [0]*3
    for train, validation in KFoldExp2.split(x):
        y_train, x_train = y[train],x[train]
        y_valid, x_valid = y[validation], x[validation]
        model.fit(y_train,x_train)
        F1, Precision, Recall = model.F1_score_with_P_and_R(y_pred =(model.predict(x_valid)), y_true = y_valid)
        print("Model ", model_num, " : ", "\nF1 Score: ", F1, "\nPrecision: ", Precision, "\nRecall: ", Recall, "\nAccuracy: ", model.accuracy(x_valid,y_valid))
        Score_Holder[0] += F1
        Score_Holder[1] += Precision
        Score_Holder[2] += Recall
        model_num += 1
    print("Average Scores: ", "\nF1 Score:", (Score_Holder[0]/5), "\nPrecision : ", (Score_Holder[1]/5), "\nRecall: ", (Score_Holder[2]/5))

def Exp3():
    #Setting up X, Y and the base model
    x = X
    y = Y
    
    model = Softmax_Logistic_Regression(0.1,0.1,100,3,0.1,1,False) #Using smaller than normal num of iteratinons for speed

    TestVector = np.zeros((7,3))
    TrainVector = np.zeros((7,3))
    DataPercent = [0.2,0.3,0.4,0.5,0.6,0.7,0.8]
    current = 0

    
    for i in DataPercent:

        n_test = int(len(y)*i) #Isolate i of dataset for training

        inds = np.random.permutation(len(y))

        Y_Train, X_Train = y[inds[:n_test]], x[inds[:n_test]]

        Y_Test, X_Test = y[inds[n_test:]], x[inds[n_test:]]

        print(len(Y_Train))
        print(len(Y_Test))

        model.fit(Y=Y_Train, X= X_Train)

        print("\n",i ," of Dataset: ")

        #Training Data
        F1, Precision, Recall = model.F1_score_with_P_and_R(y_pred = model.predict(X_Train), y_true= Y_Train)
        print("\nTraining Data: \nF1_score: ", F1, "\nPrecison:", Precision, "\nRecall: ", Recall, "\nAccuracy: ", model.accuracy(X_Train,Y_Train))
        TrainVector[current] = F1

        #Testing Data
        F1, Precision, Recall = model.F1_score_with_P_and_R(y_pred = model.predict(X_Test), y_true= Y_Test)
        print("\nTesting Data: \nF1_score: ", F1, "\nPrecison:", Precision, "\nRecall: ", Recall, "\nAccuracy: ", model.accuracy(X_Test,Y_Test))
        TestVector[current] = F1

        current += 1

    plt.plot(DataPercent, TrainVector[:,0], 'b') #Blue for Class 1
    plt.plot(DataPercent, TrainVector[:,1], 'r') # Red for Class 2
    plt.plot(DataPercent, TrainVector[:,2], 'g') #Green for Class 3
    plt.plot(DataPercent, np.mean(TrainVector, axis = 1), 'y') #Yellow is for the average
    plt.legend(['Class 1','Class 2','Class 3','Average'],loc = 'upper right')

    plt.xlabel('Proportion of Dataset')
    plt.ylabel(r'$F1 Score$')
    plt.title('F1 Score and Proportion of Dataset for Training Data')
    plt.show()

    plt.plot(DataPercent, TestVector[:,0], 'b') #Blue for Class 1
    plt.plot(DataPercent, TestVector[:,1], 'r') # Red for Class 2
    plt.plot(DataPercent, TestVector[:,2], 'g') #Green for Class 3
    plt.plot(DataPercent, np.mean(TestVector, axis = 1), 'y') #Yellow is for the average
    plt.legend(['Class 1','Class 2','Class 3','Average'],loc = 'upper right')

    plt.xlabel('Proportion of Dataset')
    plt.ylabel(r'$F1 Score$')
    plt.title('F1 Score and Proportion of Dataset for Testing Data')
    plt.show()



def Exp4():
    y = Y
    x = X

    iter_vector = [0]*5
    
    j = 0

    n_test = int(len(y)*0.2) #Isolate 20% of dataset for testing

    inds = np.random.permutation(len(y))

    Y_Test, X_Test = y[inds[:n_test]], x[inds[:n_test]]
    Y_Train, X_Train = y[inds[n_test:]], x[inds[n_test:]]

    for i in [8,16,32,64,128]:

        model = Softmax_Logistic_Regression(0.1, 0.1, 100, 3, 0.1, i, True, False)
        iter = model.fit(Y_Train,X_Train)
        F1, Precision, Recall = model.F1_score_with_P_and_R(y_pred = model.predict(X_Test), y_true= Y_Test)
        print("Model for ",i," Minibatches: ", "\nF1 Score: ", F1, "\nPrecision: ", Precision, "\nRecall: ", Recall, "\nAccuracy: ", model.accuracy(X_Test,Y_Test))
        print("Num of Iterations: ", iter)
        iter_vector[j] = iter
        j+=1

    plt.plot([8,16,32,64,128],iter_vector)
    plt.title('Number of Iterations and Minibatch Size')
    plt.xlabel('Minibatch Size')
    plt.ylabel(r'$Iterations$')
    plt.show()

def Exp5():
    x = X
    y = Y
    n_test = int(len(y)*0.2) #Isolate 20% of dataset for testing

    inds = np.random.permutation(len(y))

    Y_Test, X_Test = y[inds[:n_test]], x[inds[:n_test]]
    Y_Train, X_Train = y[inds[n_test:]], x[inds[n_test:]]

    for i in [10,1,0.1]:

        model = Softmax_Logistic_Regression(i, 0.1, 100, 3, 0.1,1,False)
        iter = model.fit(Y_Train, X_Train)
        F1, Precision, Recall = model.F1_score_with_P_and_R(y_pred = model.predict(X_Test), y_true= Y_Test)
        print("\nModel for Learning Rate ", i , " :", "\nF1 Score: ", F1, "\nPrecision: ", Precision, "\nRecall: ", Recall, "\nAccuracy: ", model.accuracy(X_Test,Y_Test))
        print("Num of Iterations: ", iter)

def Exp6():
    x = X
    y = Y
    reg_set = [0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
    iter_set = np.zeros((7,1))
    j = 0

    n_test = int(len(y)*0.2) #Isolate 20% of dataset for testing

    inds = np.random.permutation(len(y))

    Y_Test, X_Test = y[inds[:n_test]], x[inds[:n_test]]


    Y_Train, X_Train = y[inds[n_test:]], x[inds[n_test:]]

    for i in reg_set:
        model = Softmax_Logistic_Regression(0.1,0.1,100,3,i,1,False)
        iter = model.fit(Y_Train, X_Train)
        F1, Precision, Recall = model.F1_score_with_P_and_R(y_pred = model.predict(X_Test), y_true= Y_Test)
        print("\nModel for Regularization Constant", i , ":", "\nF1 Score: ", F1, "\nPrecision: ", Precision, "\nRecall: ", Recall, "\nAccuracy: ", model.accuracy(X_Test,Y_Test))
        print("Number of Iterations: ", iter)
        iter_set[j] = np.mean(F1)
        j += 1

    plt.plot(reg_set, iter_set)
    plt.xlabel('Regulariation Constant')
    plt.ylabel(r'$F1 Score$')
    plt.title('Reuglarization Constant and F1 Score')
    plt.show()
def runExp():
    print("\nExperiment 1: 80/20 Training/Test Split")
    Exp1()
    print("\nExperiment 2: 5-Fold Cross-Validation Set Implementation")
    Exp2()
    print("\nExperiment 3: Growing Training Set")
    Exp3()
    print("\nExperiment 4: SGD for Growing Minibatch Size")
    Exp4()
    print("\nExperiment 5: 3 Different Learning Rates")
    Exp5()
    print("\nExperiment 6: Optimal Parameter Choice")
    Exp6()

runExp()


            
        


