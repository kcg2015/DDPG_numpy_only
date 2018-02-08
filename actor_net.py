import numpy as np

class ActorNet(object):
  """
  A Three-layer fully-connected neural network for actor network. The net has an 
  input dimension of (N, D), with D being the cardinality of the state space.
  There are two hidden layers, with dimension of H1 and H2, respectively.  The output
  provides an action vetcor of dimenson of A. The network uses a ReLU nonlinearity
  for the first and second layer and uses tanh (scaled by a factor of
  ACTION_BOUND) for the final layer. In summary, the network has the following
  architecture:
  
  input - fully connected layer - ReLU - fully connected layer - RelU- fully co-
  nected layer - tanh*ACTION_BOUND
 """

  def __init__(self, input_size, hidden_size1, hidden_size2,output_size, std=5e-1):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H1)
    b1: First layer biases; has shape (H1,)
    W2: Second layer weights; has shape (H1, H2)
    b2: Second layer biases; has shape (H2,)
    W3: Third layer weights, has shape (H2, A)
    b3: Third layer biases; has shape  (A,)
    
    We also have the weights for a traget network (same architecture but 
    different weights)
    W1_tgt: First layer weights; has shape (D, H1)
    b1_tgt: First layer biases; has shape (H1,)
    W2_tgt: Second layer weights; has shape (H1, H2)
    b2_tgt: Second layer biases; has shape (H2,)
    W3_tgt: Third layer weights, has shape (H2, A)
    b3_tgt: Third layer biases; has shape  (A,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The continuous variables that constitutes an action vector
      of A dimension.
    """
    print("An actor network is created.")
    
    
    self.params = {}
    self.params['W1'] = self._uniform_init(input_size, hidden_size1)
    self.params['b1'] = np.zeros(hidden_size1)
    self.params['W2'] = self._uniform_init(hidden_size1, hidden_size2)
    self.params['b2'] = np.zeros(hidden_size2)
    self.params['W3'] = np.random.uniform(-3e-3, 3e-3, (hidden_size2, output_size))
    #self.params['b3'] = np.random.uniform(-3e-3, 3e-3, output_size)
    self.params['b3'] = np.zeros(output_size)
    # Initialization based on "Continuous control with deep reinformcement 
    # learning"
#    self.params['W1_tgt'] = self.params['W1']
#    self.params['b1_tgt'] = self.params['b1']
#    self.params['W2_tgt'] = self.params['W2']
#    self.params['b2_tgt'] = self.params['b2']
#    self.params['W3_tgt'] = self.params['W3']
#    self.params['b3_tgt'] = self.params['b3']
    
    
    self.params['W1_tgt'] = self._uniform_init(input_size, hidden_size1)
    self.params['b1_tgt'] = np.zeros(hidden_size1)
    self.params['W2_tgt'] = self._uniform_init(hidden_size1, hidden_size2)
    self.params['b2_tgt'] = np.zeros(hidden_size2)
    self.params['W3_tgt'] = np.random.uniform(-3e-3, 3e-3, (hidden_size2, output_size))
    self.params['b3_tgt'] = np.zeros(output_size)
    
    self.optm_cfg ={}
    self.optm_cfg['W1'] = None
    self.optm_cfg['b1'] = None
    self.optm_cfg['W2'] = None
    self.optm_cfg['b2'] = None
    self.optm_cfg['W3'] = None
    self.optm_cfg['b3'] = None
    


  def evaluate_gradient(self, X, action_grads, action_bound, target=False):
    """
    Compute the action and gradients for the network based on the input X
    
    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - target: use default weights if False; otherwise use target weights.
    - action_grads: the gradient output from the critic-network.
    - action_bound: the scaling factor for the action, which is environment
                    dependent.

    Returns:
     A tuple of:
    - actions: a continuous vector
    - grads: Dictionary mapping parameter names to gradients of those parameters; 
      has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    if not target:
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
    else:
        W1, b1 = self.params['W1_tgt'], self.params['b1_tgt']
        W2, b2 = self.params['W2_tgt'], self.params['b2_tgt']
        W3, b3 = self.params['W3_tgt'], self.params['b3_tgt']
        
    batch_size, _ = X.shape

    # Compute the forward pass
  
    scores = None
    z1=np.dot(X,W1)+b1
    H1=np.maximum(0,z1) #Activation first layer
    z2=np.dot(H1,W2)+b2
    H2=np.maximum(0,z2) #Activation second layer
    scores=np.dot(H2, W3)+b3
    actions=np.tanh(scores)*action_bound
    
    # Backward pass: compute gradients
    grads = {}
    # The derivatve at the output. Note that the derivative of tanh(x) 
    #is 1-tanh(x)**2.
    grad_output=action_bound*(1-np.tanh(scores)**2)*(-action_grads)
    # Back-propagate to second hidden layer
    out1=grad_output.dot(W3.T)
    # derivative of the max() gate
    out1[z2<=0]=0
    # Backpropagate to the first hidden layer
    out2=out1.dot(W2.T)
    # derivtive of the max() gate again
    out2[z1<=0]=0
    
    # Calculate gradient using back propagation
    grads['W3']=np.dot(H2.T, grad_output)/batch_size
    grads['W2']=np.dot(H1.T, out1)/batch_size
    grads['W1']=np.dot(X.T, out2)/batch_size
    grads['b3']=np.sum(grad_output, axis=0)/batch_size
    grads['b2']=np.sum(out1, axis=0)/batch_size
    grads['b1']=np.sum(out2, axis=0)/batch_size
                    
    return actions, grads

  def train(self, X, action_grads, action_bound):
    """
    Train this neural network using adam optimizer.
    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    """
 
     # Compute out and gradients using the current minibatch
    _, grads = self.evaluate_gradient(X, action_grads, \
                                      action_bound)
    # Update the weights using adam optimizer
    
    self.params['W3'] = self._adam(self.params['W3'], grads['W3'], config=self.optm_cfg['W3'])[0]
    self.params['W2'] = self._adam(self.params['W2'], grads['W2'], config=self.optm_cfg['W2'])[0]
    self.params['W1'] = self._adam(self.params['W1'], grads['W1'], config=self.optm_cfg['W1'])[0]
    self.params['b3'] = self._adam(self.params['b3'], grads['b3'], config=self.optm_cfg['b3'])[0]
    self.params['b2'] = self._adam(self.params['b2'], grads['b2'], config=self.optm_cfg['b2'])[0]
    self.params['b1'] = self._adam(self.params['b1'], grads['b1'], config=self.optm_cfg['b1'])[0]
    
    # Update the configuration parameters to be used in the next iteration
    self.optm_cfg['W3'] = self._adam(self.params['W3'], grads['W3'], config=self.optm_cfg['W3'])[1]
    self.optm_cfg['W2'] = self._adam(self.params['W2'], grads['W2'], config=self.optm_cfg['W2'])[1]
    self.optm_cfg['W1'] = self._adam(self.params['W1'], grads['W1'], config=self.optm_cfg['W1'])[1]
    self.optm_cfg['b3'] = self._adam(self.params['b3'], grads['b3'], config=self.optm_cfg['b3'])[1]
    self.optm_cfg['b2'] = self._adam(self.params['b2'], grads['b2'], config=self.optm_cfg['b2'])[1]
    self.optm_cfg['b1'] = self._adam(self.params['b1'], grads['b1'], config=self.optm_cfg['b1'])[1]

    
    
  
  def train_target(self, tau):
    """
      Update the weights of the target network.
     -tau: coefficent for tracking the learned network.
     """
    self.params['W3_tgt'] = tau*self.params['W3']+(1-tau)*self.params['W3_tgt']
    self.params['W2_tgt'] = tau*self.params['W2']+(1-tau)*self.params['W2_tgt']
    self.params['W1_tgt'] = tau*self.params['W1']+(1-tau)*self.params['W1_tgt']
        
    self.params['b3_tgt'] = tau*self.params['b3']+(1-tau)*self.params['b3_tgt']
    self.params['b2_tgt'] = tau*self.params['b2']+(1-tau)*self.params['b2_tgt']
    self.params['b1_tgt'] = tau*self.params['b1']+(1-tau)*self.params['b1_tgt']

      

  def predict(self, X, action_bound, target=False):
    """
    Use the trained weights of this network to predict the action vector for a 
    given state.

    Inputs:
    - X: A numpy array of shape (N, D) 
    - target: if False, use normal weights, otherwise use learned weight.
    - action_bound: the scaling factor for the action, which is environment
                    dependent.

    Returns:
    - y_pred: A numpy array of shape (N,) 
    
    """
    y_pred = None
    
    if not target:
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
    else:
        W1, b1 = self.params['W1_tgt'], self.params['b1_tgt']
        W2, b2 = self.params['W2_tgt'], self.params['b2_tgt']
        W3, b3 = self.params['W3_tgt'], self.params['b3_tgt']

    H1=np.maximum(0,X.dot(W1)+b1)
    H2=np.maximum(0,H1.dot(W2)+b2)
    scores=H2.dot(W3)+b3
    #print "scores=:", scores
    y_pred=np.tanh(scores)*action_bound
    

    return y_pred

  def _adam(self, x, dx, config=None):
      """
      Uses the Adam update rule, which incorporates moving averages of both the
      gradient and its square and a bias correction term.
    
      config format:
      - learning_rate: Scalar learning rate.
      - beta1: Decay rate for moving average of first moment of gradient.
      - beta2: Decay rate for moving average of second moment of gradient.
      - epsilon: Small scalar used for smoothing to avoid dividing by zero.
      - m: Moving average of gradient.
      - v: Moving average of squared gradient.
      - t: Iteration number (time step)
      """
      if config is None: config = {}
      config.setdefault('learning_rate', 1e-4)
      config.setdefault('beta1', 0.9)
      config.setdefault('beta2', 0.999)
      config.setdefault('epsilon', 1e-8)
      config.setdefault('m', np.zeros_like(x))
      config.setdefault('v', np.zeros_like(x))
      config.setdefault('t', 0)
      
      next_x = None
      
      #Adam update formula,                                                 #
      config['t'] += 1
      config['m'] = config['beta1']*config['m'] + (1-config['beta1'])*dx
      config['v'] = config['beta2']*config['v'] + (1-config['beta2'])*(dx**2)
      mb = config['m'] / (1 - config['beta1']**config['t'])
      vb = config['v'] / (1 - config['beta2']**config['t'])
    
      next_x = x - config['learning_rate'] * mb / (np.sqrt(vb) + config['epsilon'])
      return next_x, config
  def _uniform_init(self, input_size, output_size):
      u = np.sqrt(6./(input_size+output_size))
      return np.random.uniform(-u, u, (input_size, output_size))