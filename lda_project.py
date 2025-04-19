# %%
import pylab as pl
import scipy as sp
import numpy as np
from scipy.linalg import eig
from scipy.io import loadmat
import pdb

# %%
def load_data(fname):
    # load the data
    data = loadmat(fname)
    X,Y = data['X'],data['Y']
    # collapse the time-electrode dimensions
    X = np.reshape(X,(X.shape[0]*X.shape[1],X.shape[2]))
    # transform the labels to (-1,1)
    Y = np.sign((Y[0,:]>0) -.5)
    return X,Y

# %%
X,Y = load_data('bcidata.mat')
print(X.shape)
print(Y.shape)


# %%
def train_ncc(X,Y):
    '''
    Train a nearest centroid classifier
    '''
    X_pstv = X[:, Y == 1]
    X_ngtv = X[:, Y == -1]

    mean_pstv = X_pstv.mean(axis=1)  
    mean_ngtv = X_ngtv.mean(axis=1)

    w = mean_pstv - mean_ngtv

    b = 0.5 * np.dot(w, mean_pstv + mean_ngtv)

    # return the weight vector, bias term
    return w,b

# %%
def train_lda(X,Y):
    '''
    Train a linear discriminant analysis classifier
    '''
    X_pstv = X[:, Y == 1]  
    X_ngtv = X[:, Y == -1]

    mean_pstv = X_pstv.mean(axis=1).reshape(-1,1)  
    mean_ngtv = X_ngtv.mean(axis=1).reshape(-1,1)

    sw = np.cov(X_pstv, bias=True) + np.cov(X_ngtv, bias=True)

    w = np.linalg.pinv(sw).dot(mean_pstv - mean_ngtv).flatten()

    b = 0.5 * np.dot(w, (mean_pstv + mean_ngtv).flatten())

    # return the weight vector, bias term
    return w,b

# %%
def compare_classifiers():
    '''
    compares nearest centroid classifier and linear discriminant analysis
    '''
    fname = 'bcidata.mat'
    X,Y = load_data(fname)

    permidx = np.random.permutation(np.arange(X.shape[-1]))
    trainpercent = 70.
    stopat = int(np.floor(Y.shape[-1]*trainpercent/100.))
    #pdb.set_trace()
    
    X,Y,Xtest,Ytest = X[:,permidx[:stopat]],Y[permidx[:stopat]],X[:,permidx[stopat:]],Y[permidx[stopat:]]

    w_ncc,b_ncc = train_ncc(X,Y)
    w_lda,b_lda = train_lda(X,Y)
    fig = pl.figure(figsize=(12,5))

    ax1 = fig.add_subplot(1,2,1)
    #pl.hold(True)
    ax1.hist(w_ncc.dot(Xtest[:,Ytest<0]))
    ax1.hist(w_ncc.dot(Xtest[:,Ytest>0]))
    ax1.set_xlabel('$w^{T}_{NCC}X$')
    ax1.legend(('non-target','target'))
    ax1.set_title("NCC Acc " + str(np.sum(np.sign(w_ncc.dot(Xtest)-b_ncc)==Ytest)*100/Xtest.shape[-1]) + "%")
    ax2 = fig.add_subplot(1,2,2)
    ax2.hist(w_lda.dot(Xtest[:,Ytest<0]))
    ax2.hist(w_lda.dot(Xtest[:,Ytest>0]))
    ax2.set_xlabel('$w^{T}_{LDA}X$')
    ax2.legend(('non-target','target'))
    ax2.set_title("LDA Acc " + str(np.sum(np.sign(w_lda.dot(Xtest)-b_lda)==Ytest)*100/Xtest.shape[-1]) + "%")
    pl.savefig('ncc-lda-comparison.pdf')


# %%
compare_classifiers()

# %%
def crossvalidate(X, Y, f=10, trainfunction=train_lda):
    ''' 
    Test generalization performance of a linear classifier
    Input:  X       data (dims-by-samples)
            Y       labels (1-by-samples)
            f       number of cross-validation folds
            trainfunction   trains linear classifier
    '''
    perm = np.random.permutation(X.shape[1])
    X, Y = X[:, perm], Y[perm]

    acc_train = np.zeros(f)
    acc_test = np.zeros(f)
    folds = X.shape[1] // f

    for ifold in np.arange(f):

        test_start = ifold * folds
        test_end = (ifold + 1) * folds
        test_indices = np.arange(test_start, test_end)

        train = np.setdiff1d(np.arange(X.shape[1]), test_indices)

        # train classifier
        w, b = trainfunction(X[:, train], Y[train])  
        # compute accuracy on training data
        Y_train_pred = np.sign(np.dot(w.T, X[:, train]) - b)
        acc_train[ifold] = np.mean(Y_train_pred == Y[train])
        # compute accuracy on test data
        Y_test_pred = np.sign(np.dot(w.T, X[:, test_indices]) - b)
        acc_test[ifold] = np.mean(Y_test_pred == Y[test_indices])

    return acc_train, acc_test


# %%
X,Y = load_data('bcidata.mat')
crossvalidate(X,Y,f=10,trainfunction=train_lda)

acc_train, acc_test = crossvalidate(X, Y, f=10, trainfunction=train_lda)

# Plotting figure 2 
import matplotlib.pyplot as plt

data = [acc_train, acc_test]
plt.figure(figsize=(8, 6))
boxprops = dict(color='blue')
whiskerprops = dict(color='blue', linestyle='--')
capprops = dict(color='blue')
medianprops = dict(color='red')
flierprops = dict(markerfacecolor='blue', marker='o', linestyle='none')

plt.boxplot(data,
            patch_artist=False,
            boxprops=boxprops,
            whiskerprops=whiskerprops,
            capprops=capprops,
            medianprops=medianprops,
            flierprops=flierprops)

plt.xticks([1, 2], ['train block', 'test block'])
plt.ylabel('Accuracy')
plt.ylim(0.945, 0.985)
plt.title('Classification accuracy for training and test blocks in 10-fold cross-validation')
plt.tight_layout()
plt.savefig("figure2_boxplot_accy.pdf")
plt.show()






