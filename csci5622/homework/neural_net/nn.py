import argparse
import numpy as np 
import cPickle, gzip


class Network:
    def __init__(self, sizes):
        self.L = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(n, 1) for n in self.sizes[1:]]
        self.weights = [np.random.randn(n, m) for (m,n) in zip(self.sizes[:-1], self.sizes[1:])]
        self.train_acc = []
        self.test_acc = []
        
    def g(self, z):
        """
        activation function 
        """
        return sigmoid(z)
    
    def g_prime(self, z):
        """
        derivative of activation function 
        """
        return sigmoid_prime(z) 
    
    def forward_prop(self, a):
        """
        memory aware forward propagation for testing 
        only.  back_prop implements it's own forward_prop 
        """
        for (W,b) in zip(self.weights, self.biases):
            a = self.g(np.dot(W, a) + b) 
        return a
    
    def gradC(self, a, y):
        """
        gradient of cost function 
        Assumes C(a,y) = (a-y)^2/2 
        """
        return (a - y)
    
    def SGD_train(self, train, epochs, eta, lam=0.0, verbose=True, test=None):
        """
        SGD for training parameters  
        epochs is the number of epocs to run 
        eta is the learning rate 
        lam is the regularization parameter 
        If verbose is set will print progressive accuracy updates 
        If test set is provided, routine will print accuracy on test set as learning evolves
        """
        n_train = len(train)
        for epoch in range(epochs):
            perm = np.random.permutation(n_train)
            for kk in range(n_train):
                xk = train[perm[kk]][0]
                yk = train[perm[kk]][1]
                dWs, dbs = self.back_prop(xk, yk)
                #l2-regularized --- Update rule: (1 - 2 * eta * lam * .5) = (1 - eta * lam)
                
                self.weights = [(1 - eta*lam) * W - eta*dW for (W, dW) in zip(self.weights, dWs)]
                self.biases = [b - eta*db for (b, db) in zip(self.biases, dbs)]

            if verbose:
                if epoch==0 or (epoch + 1) % 50 == 0:
                    acc_train = self.evaluate(train)
                    self.train_acc.append(acc_train)
                    if test is not None:
                        acc_test = self.evaluate(test)
                        self.test_acc.append(acc_test)
                        print "Epoch {:4d}: Train {:10.5f}, Test {:10.5f}".format(epoch+1, acc_train, acc_test) 
                    else:
                        print "Epoch {:4d}: Train {:10.5f}".format(epoch+1, acc_train)    
                
    def back_prop(self, x, y):
        """
        Back propagation for derivatives of C wrt parameters 
        """
        db_list = [np.zeros(b.shape) for b in self.biases]
        dW_list = [np.zeros(W.shape) for W in self.weights]
        
        a = x 
        a_list = [a]
        z_list = [np.zeros(a.shape)] # Pad with throwaway so indices match 
        
        for W, b in zip(self.weights, self.biases):
            z = np.dot(W, a) + b 
            z_list.append(z)
            a = self.g(z)
            a_list.append(a)
            
        # Back propagate deltas to compute derivatives 

        delta = self.gradC(a_list[-1], y) * self.g_prime(z_list[-1])

        for ell in range(self.L-2,-1,-1):
            db_list[ell] = delta
            dW_list[ell] = np.dot(delta, a_list[ell].transpose())
            delta = np.dot(self.weights[ell].transpose(), delta) * self.g_prime(z_list[ell])

        return (dW_list, db_list)
    
    def evaluate(self, test):
        """
        Evaluate current model on labeled test data 
        """
        ctr = 0 
        for x, y in test:
            yhat = self.forward_prop(x)
            ctr += np.argmax(yhat) == np.argmax(y)
        return float(ctr) / float(len(test))
    
    def compute_cost(self, x, y):
        """
        Evaluate the cost function for a specified 
        training example. 
        """
        a = self.forward_prop(x)
        return 0.5*np.linalg.norm(a-y)**2
    
    def gradient_checking(self, train, EPS=0.0001):
        """
        Performs gradient checking on all weights in the 
        network for a randomly selected training example 
        """
        # Randomly select a training example 
        kk = np.random.randint(0,len(train))
        xk = train[kk][0]
        yk = train[kk][1]

        # Get the analytic(ish) weights from back_prop 
        dWs, dbs = self.back_prop(xk, yk)

        # List of relative errors.  Used only for unit testing. 
        rel_errors = []

        # Loop over and perturb each weight/bias in 
        # network and test numerical derivative 
        # Don't forget that after perturbing the weights
        # you'll want to put them back the way they were! 
        
        # Loop over and perturb each weight/bias in 
        # network and test numerical derivative 
        for ell in range(self.L-1):
            for ii in range(self.weights[ell].shape[0]):
                # Check weights in level W[ell][ii,jj] 
                for jj in range(self.weights[ell].shape[1]):
                    true_dW = dWs[ell][ii][jj]
                    
                    temp = self.weights[ell][ii][jj]

                    self.weights[ell][ii][jj] += EPS 
                    loss1 = self.compute_cost(xk, yk)
                    
                    self.weights[ell][ii][jj] = temp
                    
                    self.weights[ell][ii][jj] -= EPS   
                    loss2 = self.compute_cost(xk, yk)
                    
                    num_dW = (loss1 - loss2) / (2 * EPS)
                    
                    self.weights[ell][ii][jj] = temp

                    rel_dW = np.abs(true_dW-num_dW)/np.abs(true_dW)
                    print "w: {: 12.10e}  {: 12.10e} {: 12.10e}".format(true_dW, num_dW, rel_dW)
                    rel_errors.append(rel_dW)


                true_db = dbs[ell][ii][0]
            
                temp = self.biases[ell][ii][0]
                
                self.biases[ell][ii][0] += EPS
                loss1 = self.compute_cost(xk, yk)
                
                self.biases[ell][ii][0] = temp
                
                self.biases[ell][ii][0] -= EPS
                loss2 = self.compute_cost(xk, yk)
                
                num_db = (loss1 - loss2) / (2 * EPS)
                
                self.biases[ell][ii][0] = temp
                
                rel_db = np.abs(true_db-num_db)/np.abs(true_db)
                print "b: {: 12.10e}  {: 12.10e} {: 12.10e}".format(true_db, num_db, rel_db)
                rel_errors.append(rel_db)

        return rel_errors
                
def sigmoid(z, threshold=20):
    z = np.clip(z, -threshold, threshold)
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1.0 - sigmoid(z))



if __name__ == "__main__":
    
    f = gzip.open('../data/tinyTOY.pkl.gz', 'rb') # change path to ../data/tinyMNIST.pkl.gz after debugging
    train, test = cPickle.load(f)

    nn = Network([2,100,2])
    nn.SGD_train(train, epochs=500, eta=.25, lam=0.0, verbose=True, test=test)
    

    f = gzip.open('../data/tinyMNIST.pkl.gz', 'rb')

    train, test = cPickle.load(f) 

    nn = Network([196,100,10])
    nn.SGD_train(train, epochs=100, eta=0.25, lam=0.001, verbose=True, test=test)
