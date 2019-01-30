from proj1_helpers import *
import numpy as np
import random
import matplotlib.pyplot as plt

######################### Loss Functions #########################

# Compute loss with Mean Squared Error
def compute_loss(y, tx, w):
    e = y.reshape((len(y),1)) - tx.dot(w).reshape((len(y),1))
    return 1/2*np.mean(e**2)

# Compute gradient for gradient descent
def compute_gradient(y, tx, w):
    e = y.reshape((len(y),1)) - tx.dot(w).reshape((len(y),1))
    grad = -tx.T.dot(e) / len(e)
    return grad

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def compute_loss_logistic(y, tx, w):
    # loss formula works only for y = {0,1} 
    y[y == -1] = 0
    y = y.reshape((len(y),1))
    sigma = sigmoid(tx.dot(w)).reshape((len(y),1))
    loss = y.T.dot(np.log(sigma)) + (1 - y).T.dot(np.log(1 - sigma))
    return np.squeeze(- loss)

def compute_gradient_logistic(y, tx, w):
    sigma = sigmoid(tx.dot(w)).reshape((len(y),1))
    y = y.reshape((len(sigma),1))
    grad = tx.T.dot(sigma - y)
    return grad


######################### Methods Implementation #########################

# Gradient Descent
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    ws = [initial_w]
    losses = []
    w = initial_w

    for n_iter in range(max_iters):
        gradient = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)
        w = w - gamma * gradient

        ws.append(w)
        losses.append(loss)

    return ws[-1], losses[-1]

# Stochastic Gradient Descent
def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    ws = [initial_w]
    losses = []
    w = initial_w
    n_iter = 0
    batch_size = 1

    for batch_y, batch_tx in batch_iter(y, tx, batch_size, max_iters):
        grad = compute_gradient(batch_y, batch_tx, w)
        loss = compute_loss(batch_y, batch_tx, w)
        w = w - gamma * grad

        ws.append(w)
        losses.append(loss)
        n_iter += 1

    return ws[-1], losses[-1]

    
def least_squares(y, tx):
    w = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))
    loss = compute_loss(y, tx ,w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    w = np.linalg.solve(tx.T.dot(tx) + lambda_ * np.eye(tx.shape[1]), tx.T.dot(y))
    loss = compute_loss(y, tx, w)
    return w, loss

def logistic_regression(y, tx, w, max_iters, gamma):
    for n_iter in range(max_iters):
        loss = compute_loss_logistic(y, tx, w)
        grad = compute_gradient_logistic(y, tx, w)
        w -= gamma * grad
    return w, loss

def reg_logistic_regression(y, tx, lambda_, w, max_iters, gamma):
    for n_iter in range(max_iters):
        loss = compute_loss_logistic(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w))
        grad = compute_gradient_logistic(y, tx, w) + 2 * lambda_ * w
        w -= gamma * grad
    return w, loss


######################### Improvements #########################

def RR_optimal_lambda_finder(y, tx, learning_algo):
    k_folds = 10
    lambdas = np.logspace(-4,0,30)
    seeds = range(10)
    
    # define an empty matrix to store cross validation errors
    CV_errors = np.empty((len(seeds), len(lambdas)), dtype=float)
    for i, seed in enumerate(seeds):
        for j, lambda_ in enumerate(lambdas):
            errors = cross_validation(y, tx, k_folds, learning_algo, lambda_, seed)
            CV_error = np.mean(errors)
            CV_errors[i, j] = CV_error

    best_accuracy = max(np.mean(CV_errors, axis=0))
    opt_lambda = lambdas[np.argmax(np.mean(CV_errors, axis=0))]
    return opt_lambda, best_accuracy 


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly[:,1:]


def interaction_forward_selection(y, tx):
    '''For every possible 2nd order interaction term, add to the original \
    feature set iff its inclusion leads to higher accuracy based on 5-fold CV'''
    
    # define reference accuracy (with NO interaction terms)
    reference = np.mean(cross_validation(y, tx, 5, least_squares, 0, 1))
    
    # define list to store feature indices whose interaction is useful
    interaction_terms = []
    
    counter = 0 
    num_features = 30 # original number of features
    for col1 in range(num_features):
        for col2 in range(num_features):
            if col1 >= col2: continue
            temp_tx = np.c_[tx, tx[:,col1] * tx[:,col2]]
            accuracy = np.mean(cross_validation(y, temp_tx, 5, least_squares, 0, 1))
            
            # if new accuracy is higher, add the term
            if accuracy > reference: 
                reference = accuracy
                tx = temp_tx
                interaction_terms.append((col1, col2))
            
            counter += 1 
            print("{p:.2f}% complete, best accuracy: {a:.9f}".format(p=100* counter / 435, a=reference))
    return tx, interaction_terms


def third_interaction_forward_selection(y, tx):
    '''For every possible 3rd order interaction term, add to the original \
    feature set iff its inclusion leads to higher accuracy based on 5-fold CV'''
    
    # define reference accuracy (with NO interaction terms)
    reference = np.mean(cross_validation(y, tx, 5, least_squares, 0, 1))
    
    # define list to store feature indices whose interaction is useful
    third_interaction_terms = []
    
    counter = 0 # delete this line
    num_features = 30 # original number of features
    for col1 in range(num_features):
        for col2 in range(num_features):
            if col1 >= col2: continue
            for col3 in range(num_features):
                if col2 >= col3: continue
                temp_tx = np.c_[tx, tx[:,col1] * tx[:,col2] * tx[:,col3]]
                accuracy = np.mean(cross_validation(y, temp_tx, 5, least_squares, 0, 1))
            
                # if new accuracy is higher, add the term
                if accuracy > reference: 
                    reference = accuracy
                    tx = temp_tx
                    third_interaction_terms.append((col1, col2, col3))
            
                counter += 1 # delete this line
                print("{p:.2f}% complete, best accuracy: {a:.9f}".format(p=100* counter / 4060, a=reference))
    return tx, third_interaction_terms


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation(y, tx, k_folds, learning_algo, lambda_, seed):
    # build k_folds instances of indices 
    k_indices = build_k_indices(y, k_folds, seed)
    
    # define list to store cross validation error
    errors = []
    for k in range(k_folds):
        tx_valid = tx[k_indices[k,:]]
        y_valid = y[k_indices[k,:]]
        tx_train = tx[k_indices[list(set(range(k_indices.shape[0])) - set([k])),:].reshape((k_indices.shape[0]-1)*k_indices.shape[1]),:]
        y_train = y[k_indices[list(set(range(k_indices.shape[0])) - set([k])),:].reshape((k_indices.shape[0]-1)*k_indices.shape[1])]

        # least squares using normal equations
        if learning_algo == least_squares:
            w, loss_tr = learning_algo(y_train, tx_train)
            
        # ridge regression using normal equations
        elif learning_algo == ridge_regression:
            w, loss_tr = learning_algo(y_train, tx_train, lambda_)
        
        # least squares gradient descent
        elif learning_algo == least_squares_GD:
            initial_w = np.zeros((tx.shape[1],1))
            max_iters = 1000
            gamma = 0.0000001
            w, loss_tr = learning_algo(y_train, tx_train, initial_w, max_iters, gamma)
        
        # least squares stochastic gradient descent
        elif learning_algo == least_squares_SGD:
            initial_w = np.zeros((tx.shape[1],1))
            max_iters = 1000
            gamma = 0.0000001
            w, loss_tr = learning_algo(y_train, tx_train, initial_w, max_iters, gamma)
        
        # logistic regression gradient descent
        elif learning_algo == logistic_regression:
            initial_w = np.zeros((tx.shape[1],1))
            max_iters = 500
            gamma = 0.000000000000001
            w, loss_tr = learning_algo(y_train, tx_train, initial_w, max_iters, gamma)
        
        # regularized logistic regression gradient descent
        elif learning_algo == reg_logistic_regression:
            initial_w = np.zeros((tx.shape[1],1))
            max_iters = 500
            gamma = 0.000000000000001
            w, loss_tr = learning_algo(y_train, tx_train, lambda_, initial_w, max_iters, gamma)
            
        y_hat = predict_labels(w, tx_valid)
        errors.append(sum(y_valid.reshape((len(y_valid),1))==y_hat.reshape((len(y_hat),1))) / len(y_valid))
    
    # return the average error rate across the folds
    return errors

def data_segmentation(y, tx):
    '''
    PRI_jet_num is a feature which only takes a value of 0, 1, 2, or 3.
    Many features become undefined (-999) based on which value it takes.
    The purpose of this function is to split the data based on the four values.
    Source: http://opendata.cern.ch/record/328 
    Input:
        y: reponse
        tx: data matrix
    Returns:
        four sets of response and data matrices segmented based on PRI_jet_num
    '''
    # data segmentation
    temp_matrix = np.c_[y, tx]
    
    indices_0 = temp_matrix[:,23]==0
    temp_matrix_0 = temp_matrix[indices_0,:]
    y_0 = temp_matrix_0[:,0]
    tx_0 = temp_matrix_0[:,1:]
    
    indices_1 = temp_matrix[:,23]==1
    temp_matrix_1 = temp_matrix[indices_1,:]
    y_1 = temp_matrix_1[:,0]
    tx_1 = temp_matrix_1[:,1:]
    
    indices_2 = temp_matrix[:,23]==2
    temp_matrix_2 = temp_matrix[indices_2,:]
    y_2 = temp_matrix_2[:,0]
    tx_2 = temp_matrix_2[:,1:]
    
    indices_3 = temp_matrix[:,23]==3
    temp_matrix_3 = temp_matrix[indices_3,:]
    y_3 = temp_matrix_3[:,0]
    tx_3 = temp_matrix_3[:,1:]
    
    # when PRI_jet_num is 0, the following features are undefined and thus removed
    tx_0 = np.delete(tx_0, np.s_[4,5,6,12,22,23,24,25,26,27,28,29], axis=1)
    
    # when PRI_jet_num is 1, the following features are undefined and thus removed
    tx_1 = np.delete(tx_1, np.s_[4,5,6,12,22,26,27,28], axis=1)
    
    # at least, PRI_jet_num itself is removed
    tx_2 = np.delete(tx_2, np.s_[22], axis=1)
    tx_3 = np.delete(tx_3, np.s_[22], axis=1)
    
    # replace any remaining -999 values with the mean of that feature
    tx_0 = replace_with_mean(tx_0)
    tx_1 = replace_with_mean(tx_1)
    tx_2 = replace_with_mean(tx_2)
    tx_3 = replace_with_mean(tx_3)
        
    return y_0, tx_0, y_1, tx_1, y_2, tx_2, y_3, tx_3, indices_0, indices_1, indices_2, indices_3

def backward_selection(y, tx):
    '''Performs backward feature selection using least squares algorithm
    Input:
        y: response
        tx: data matrix
    Output:
        new data matrix with (potentially) fewer features'''
    
    cols_removed = []    
    temp_tx, col_removed = backward_selection_algorithm(y, tx)
    while tx.shape[1] != temp_tx.shape[1]: # means a feature was removed 
        tx = temp_tx
        cols_removed.append(col_removed)
        temp_tx, col_removed = backward_selection_algorithm(y, temp_tx)
    return tx, cols_removed

def backward_selection_algorithm(y, tx):
    k_folds = 10
    seed = 1
    index_to_remove = []
    reference = np.mean(cross_validation(y, tx, k_folds, least_squares, 0.0001, seed))
    for c in range(tx.shape[1]):
        temp_tx = tx[:,list(set(range(tx.shape[1])) - set([c]))]
        CV_accuracy = np.mean(cross_validation(y, temp_tx, k_folds, least_squares, 0.0001, seed))
        if CV_accuracy > reference:
            reference = CV_accuracy
            index_to_remove.append(c)
    
    if len(index_to_remove) == 0: # means no features were removed
        return tx, -1
    
    return tx[:,list(set(range(tx.shape[1])) - set([index_to_remove[-1]]))], index_to_remove[-1]

def replace_with_mean(tx):
    '''replace all -999 values with mean value of each column'''
    for col in range(tx.shape[1]):
        # find indices for which the value is -999
        indices = tx[:,col]==-999
        # replace with mean value
        tx[indices,col] = np.mean(tx[~indices,col])
    return tx

######################### Helpers #########################

# Creates batches for stochastic gradient descent
def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
