import numpy as np
import random

def softmax(x):
    """
    Compute the softmax function for each row of the input x.

    It is crucial that this function is optimized for speed because
    it will be used frequently in later code.
    You might find numpy functions np.exp, np.sum, np.reshape,
    np.max, and numpy broadcasting useful for this task. (numpy
    broadcasting documentation:
    http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

    You should also make sure that your code works for one
    dimensional inputs (treat the vector as a row), you might find
    it helpful for your later problems.

    You must implement the optimization in problem 1(a) of the 
    written assignment!
    """

    #print 'x', x
    #print x.shape, len(x.shape)

    if len(x.shape) == 1:

        my_max = np.max(x)
        my_exp = np.exp(x - my_max)
        my_sums = np.sum(my_exp)

        #print 'my_exp', my_exp
        #print 'my_sums', my_sums        

        res = my_exp / my_sums
    else:
        my_max = np.amax(x, axis=1)
        n_rows = my_max.shape[0]

        #print 'my_max', my_max

        my_max = my_max.reshape(n_rows, 1)

        diff = x - my_max
        #print 'diff', diff

        my_exp = np.exp(diff)

        #print 'my_exp.shape', my_exp.shape

        my_sums = np.sum(my_exp,axis=1)

        #print 'my_sums.shape', my_sums.shape

        res = (my_exp.T / my_sums).T

    return res

def test_softmax_basic():
    """
    Some simple tests to get you started. 
    Warning: these are not exhaustive.
    """
    print "Running basic tests..."
    test1 = softmax(np.array([1,2]))
    print test1
    assert np.amax(np.fabs(test1 - np.array(
        [0.26894142,  0.73105858]))) <= 1e-6

    test2 = softmax(np.array([[1001,1002],[3,4]]))
    print test2
    assert np.amax(np.fabs(test2 - np.array(
        [[0.26894142, 0.73105858], [0.26894142, 0.73105858]]))) <= 1e-6

    test3 = softmax(np.array([[-1001,-1002]]))
    print test3
    assert np.amax(np.fabs(test3 - np.array(
        [0.73105858, 0.26894142]))) <= 1e-6

    print "You should verify these results!\n"

def test_softmax():
    """ 
    Use this space to test your softmax implementation by running:
        python q1_softmax.py 
    This function will not be called by the autograder, nor will
    your tests be graded.
    """
    print "Running your tests..."
    ### YOUR CODE HERE
    raise NotImplementedError
    ### END YOUR CODE  

if __name__ == "__main__":
    test_softmax_basic()
    #test_softmax()