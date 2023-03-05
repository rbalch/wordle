import os
import torch
import torch.nn as nn

local_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out_dir = os.path.join(local_dir, 'out')


class NeuralNet(nn.Module):

    def __init__(self, key, input_size, hidden_sizes, output_size, activation='tanh'):
        super(NeuralNet, self).__init__()
        self.key = key
        self.activation = getattr(torch, activation, None) # tanh sigmoid relu
        assert self.activation is not None, f'Invalid activation function: {activation}'
        self.hidden_layers = nn.ModuleList()
        # create the hidden layers
        for hidden_size in hidden_sizes:
            self.hidden_layers.append(nn.Linear(input_size, hidden_size))
            input_size = hidden_size
        # create the output layer
        self.output_layer = nn.Linear(input_size, output_size)

    def __str__(self):
        return f'NeuralNet(key={self.key}, layers={[input_size] + hidden_sizes + [output_size]})'
    
    def forward(self, observation, store_gradients=False, activate_output=False):
        # TODO: can't decide if i want to use activation on the output layer or not

        def run(x):
            # pass the input through each hidden layer with an activation
            for hidden_layer in self.hidden_layers:
                x = self.activation(hidden_layer(x))
            return self.activation(self.output_layer(x)) if activate_output else self.output_layer(x)
        
        if store_gradients:
            return run(observation)
        else:
            # if we aren't doing any backpropagation, we don't need to keep track of the intermediate values
            with torch.no_grad():
                return run(observation)
            
    def train(self, observation, target, lr=0.01):
        # define loss function
        loss_function = nn.MSELoss()
        # create optimizer; stochastic gradient descent; lr = learning rate
        optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        # run with gradients and get loss
        prediction = self.forward(observation, store_gradients=True)
        loss = loss_function(prediction, target)
        # zero gradients, perform a backward pass, and update the weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def load(self, filename=None):
        filename = filename or os.path.join(out_dir, 'best_genome.pt')
        self.load_state_dict(torch.load(filename))

    def save(self, filename=None):
        filename = filename or os.path.join(out_dir, 'best_genome.pt')
        torch.save(self.state_dict(), filename)


if __name__ == "__main__":
    input_size = 10
    hidden_sizes = [20, 30, 40]
    output_size = 5
    
    g = NeuralNet(0, input_size, hidden_sizes, output_size)
    print(g)
    x = torch.rand(1, input_size)
    # print(x)
    y = g.forward(x)
    print(y)
    # print(torch.tanh(y))

    # for _ in range(100):
    #     g.train(x, torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]]))
    #     print(g.forward(x))

    # print('------------------')
    # for _ in g.parameters():
    #     print(_)
    #     break

    # for name, param in g.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.data)
