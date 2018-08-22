import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)
    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.
    
    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength
    
    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_class = W.shape[1]
    num_train = X.shape[0]
    
    for i in range(num_train):
        f = np.dot(X[i], W)
        f -= np.max(f) #shift the value to ensure stability
        
        soft = np.exp(f[y[i]]) / np.sum(np.exp(f))
        loss += (-1) * np.log(soft)
        
        dW[:,y[i]] -= X[i].T
        for j in range(num_class):
            dW[:,j] += np.exp(f[j]) / np.sum(np.exp(f)) * X[i].T 
        #print(f)
            
    #print("1:")
    #print(dW)
        
    loss /= num_train
    dW /= num_train
    
    loss += 0.5 * reg * np.sum(W * W)
    dW += reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.
    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_class = W.shape[1]
    num_train = X.shape[0]
    num_dim = X.shape[1]
    
    f = np.dot(X, W)
    f -= np.max(f, axis=1).reshape(num_train,1) #shift the value to ensure stability
    
    correct = np.zeros(num_train)
    for i in range(num_train):
        correct[i] = f[i, y[i]]
        
    f_sum = np.sum(np.exp(f), axis=1).reshape(num_train, 1) #shape[N, 1]
    loss += np.sum(np.log(f_sum) - correct.reshape(num_train,1))
    f_final = np.exp(f) / f_sum
  
    #solution 1
    #for j in range(num_train):
    #    dW[:,y[j]] -= X[j].T
    #    dW += np.dot(X[j].reshape(1,num_dim).T , f_final[j,:].reshape(1,num_class))
    
    #solution 2
    f_final[range(num_train),y] -= 1
    dW = np.dot(X.T, f_final)
        
    loss /= num_train
    dW /= num_train
    
    loss += 0.5 * reg * np.sum(W * W)
    dW += reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
    return loss, dW

