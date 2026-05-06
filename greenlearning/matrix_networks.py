from .neural_network import NeuralNetwork

def matrix_networks(layers, activation, shape, dropout_rate=0.0):
    """Create a matrix of neural networks with the given parameters.
    
    Example: 
    
    .. code-block:: python
    
        gl.matrix_networks([2] + [50] * 4 + [1], "rational", (2,1)) 
   
    creates a matrix size 2 x 1 of rational networks with 4 hidden layers of 50 neurons. 
    """
    
    # Initialize the array
    M = []
    
    # Create a matrix
    if len(shape) > 1:
        for i in range(shape[0]):
            Row = []
            for j in range(shape[1]):
                Row.append(NeuralNetwork(layers, activation, dropout_rate=dropout_rate))
            M.append(Row)
    # Create a vector
    else:
        for i in range(shape[0]):
            M.append(NeuralNetwork(layers, activation, dropout_rate=dropout_rate))
    
    return M
