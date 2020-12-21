import numpy as np


class NeuralNetwork(object):
    
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        
        # Setting the number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        
        # Initializing weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
                                       (self.input_nodes, self.hidden_nodes))
        
       
        
        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.output_nodes))
        
       
        # Setting the Learning Rate
        self.lr = learning_rate
      
        # Setting the Activation Function
        self.activation_function = lambda x : (1/(1+np.exp((-1)*x))) 
       

    def train(self, features, targets):
        
        #Training the Model
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        
        for X, y in zip(features, targets):
            
            final_outputs, hidden_outputs = self.forward_pass_train(X)  
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y, 
                                                                        delta_weights_i_h, delta_weights_h_o)
        #Updating Weights     
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)


    def forward_pass_train(self, X):
      
        # Signals into hidden layer
        hidden_inputs = np.dot(X,self.weights_input_to_hidden) 
        
        # Signals from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs) 
              
        # Signals into final output layer
        final_inputs = np.dot(hidden_outputs,self.weights_hidden_to_output) 
        
        # Signals from final output layer
        final_outputs = final_inputs 
        
        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        

        # Output Error 
        error = y-final_outputs
        
        #Error Terms
        output_error_term = error*1
        

        
        hidden_error_term = (output_error_term)*(self.weights_hidden_to_output.reshape(-1,1))*(hidden_outputs.reshape(-1,1))*(1-hidden_outputs.reshape(-1,1))
        
        

        # Weight step (input to hidden)
        delta_weights_i_h += self.lr * np.dot(X.reshape(-1,1),hidden_error_term.T)
      
        # Weight step (hidden to output)             
        delta_weights_h_o += (self.lr * hidden_outputs * output_error_term).reshape(-1,1)

        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
         
        #Updating Weights for Gradient Descent Steps
        self.weights_hidden_to_output += delta_weights_h_o
        self.weights_input_to_hidden += delta_weights_i_h 

        
    def run(self, features):
        
        # signals into hidden layer
        hidden_inputs = np.dot(features,self.weights_input_to_hidden) 
        # signals from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs) 
        
        # signals into final output layer
        final_inputs = np.dot(hidden_outputs,self.weights_hidden_to_output) 
        # signals from final output layer 
        final_outputs = final_inputs 
        
        return final_outputs




