import os
import numpy as np
from scipy.io import loadmat

# set seet
np.random.seed(0)

def split_data(data, gt, split=[0.5, 0.2, 0.3]): 
    X_Train = None
    X_Val = None
    for i in range(int(np.max(np.unique(gt)))): 

        val = (gt == i + 1)
        val = np.reshape(val, np.shape(data)[0])
        firmas = data[val]
        Len = firmas.shape[0]
        pp = np.random.permutation(Len)
        if X_Train is not None:

            temporal = firmas[pp[0:int(Len * split[0])]]
            X_Train = np.concatenate([X_Train, temporal], axis=0)
            Y_Train = np.concatenate([Y_Train, np.ones((temporal.shape[0], 1)) * i])

            temporal = firmas[pp[int(Len * np.sum(split[0])):int(Len * np.sum(split[0:2]))]]
            X_Val = np.concatenate([X_Val, temporal], axis=0)
            Y_Val = np.concatenate([Y_Val, np.ones((temporal.shape[0], 1)) * i])

            temporal = firmas[pp[int(Len * np.sum(split[0:2])):int(Len * np.sum(split[0:3]))]]
            X_Test = np.concatenate([X_Test, temporal], axis=0)
            Y_Test = np.concatenate([Y_Test, np.ones((temporal.shape[0], 1)) * i])

        else:
            X_Train = firmas[pp[0:int(Len * split[0])]]
            Y_Train = np.ones((X_Train.shape[0], 1)) * i

            X_Val = firmas[pp[int(Len * np.sum(split[0])):int(Len * np.sum(split[0:2]))]]
            Y_Val = np.ones((X_Val.shape[0], 1)) * i

            X_Test = firmas[pp[int(Len * np.sum(split[0:2])):int(Len * np.sum(split[0:3]))]]
            Y_Test = np.ones((X_Test.shape[0], 1)) * i

    p1 = np.random.permutation(len(X_Train))
    p2 = np.random.permutation(len(X_Val))
    p3 = np.random.permutation(len(X_Test))

    X_Train = X_Train[p1]
    Y_Train = Y_Train[p1]
    X_Val = X_Val[p2]
    Y_Val = Y_Val[p2]
    X_Test = X_Test[p3]
    Y_Test = Y_Test[p3]


    # Y_Train = np.argmax(Y_Train, axis=1)
    # Y_Val = np.argmax(Y_Val, axis=1)
    # Y_Test = np.argmax(Y_Test, axis=1)

    return X_Train, Y_Train, X_Val, Y_Val, X_Test, Y_Test


if __name__ == "__main__":

    dataset = "indian_pines"
    data = loadmat(os.path.join("data", dataset, f"{dataset}_corrected.mat"))["hyperimg"]
    gt   = loadmat(os.path.join("data", dataset, f"{dataset}_gt.mat"))["hyperimg_gt"]

    data = data.reshape(-1, data.shape[-1])
    gt = gt.reshape(-1)

    X_Train, Y_Train, X_Val, Y_Val, X_Test, Y_Test = split_data(data, gt, split=[0.1, 0.1, 0.8])

    
    np.save(os.path.join("data", dataset, "Xtrain.npy"), X_Train)
    np.save(os.path.join("data", dataset, "Ytrain.npy"), Y_Train)
    np.save(os.path.join("data", dataset, "Xval.npy"), X_Val)
    np.save(os.path.join("data", dataset, "Yval.npy"), Y_Val)
    np.save(os.path.join("data", dataset, "Xtest.npy"), X_Test)
    np.save(os.path.join("data", dataset, "Ytest.npy"), Y_Test)
