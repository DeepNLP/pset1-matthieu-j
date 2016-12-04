import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive

def forward_backward_prop(data, labels, params, dimensions):
    """ 
    Forward and backward propagation for a two-layer sigmoidal network 
    
    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    ### YOUR CODE HERE: forward propagation
    
    #data : 20x10, W1 = 10x5
    z2_0 = np.dot(data, W1) #20x5
    z2 = np.add(z2_0, b1) #20x5

    #print 'z2_0', z2_0
    #print 'b1', b1
    #print 'z2', z2

    a2 = sigmoid(z2) #20x5

    z3_0 = np.dot(a2, W2)
    z3 = np.add(z3_0, b2)

    a3 = sigmoid(z3) #20x10

    diffs = (-1)*labels*np.log(a3)
    

    #print 'b2', b2, b2.shape
    #print 'diff', diff

    cost = np.sum(diffs)

    #loc_cost = 0.5 * np.square(sigmoid(b2[0,9]))

    #print 'b2', b2[0,9]
    #loc_cost = sigmoid(b2[0,9])
    #cost = loc_cost

    #print 'cost', cost, 'grad', sigmoid_grad(b2[0,9])

    ### END YOUR CODE
    
    ### YOUR CODE HERE: backward propagation

    delta_3 = (-1) * np.multiply(labels, sigmoid_grad(a3) / a3)

    gradb2 = np.sum(delta_3, axis=0)

    #print 'delta_3', delta_3, len(delta_3)
    #print 'gradb2', gradb2.shape

    gradW2 = np.dot(a2.T, delta_3)

    #print 'gradW2', gradW2, gradW2.shape

    #print 'W2.shape', W2.shape, delta_3.shape

    delta_2_0 = np.dot(delta_3, W2.T)
    delta_2 = np.multiply(delta_2_0, sigmoid_grad(a2))

    gradW1 = np.dot(data.T, delta_2)

    gradb1 = np.sum(delta_2, axis=0)

    #print 'gradb1', gradb1.shape

    #raise NotImplementedError
    ### END YOUR CODE
    
    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(), 
        gradW2.flatten(), gradb2.flatten()))

    #print 'shape', grad.shape
    #print 'cost', cost
    
    return cost, grad

def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using 
    gradcheck.
    """
    print "Running sanity check..."

    #np.random.seed(0)

    N = 1
    dimensions = [10, 5, 10]

    data = np.random.randn(N, dimensions[0])   # each row will be a datum

    #print 'data', data

    labels = np.zeros((N, dimensions[2]))

    #random.seed(0)

    for i in xrange(N):
        labels[i,random.randint(0,dimensions[2]-1)] = 1

    #print 'labels', labels

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    #print 'params', params

    f = lambda params: forward_backward_prop(data, labels, params, dimensions)
    
    gradcheck_naive(f, params)

    return

def your_sanity_checks(): 
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py 
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print "Running your sanity checks..."
    
    x = -0.694567859731
    h = 1.0e-6

    fxplush = sigmoid(x + h)
    fxminush = sigmoid(x - h)

    fx = sigmoid(x)

    print 'numgrad', (fxplush - fxminush) / (2*h)

    print 'real grad', sigmoid_grad(fx)
    #print 'fx - fx2', fx - fx*fx

if __name__ == "__main__":
    sanity_check()
    #your_sanity_checks()