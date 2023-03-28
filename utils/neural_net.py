import os
import importlib
import pickle
import torch
import torch.nn as nn
import uuid
from copy import deepcopy

local_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out_dir = os.path.join(local_dir, 'out')


class NeuralNet(nn.Module):

    def __init__(self, key, input_size, hidden_sizes, output_size,
                    activation=None,
                    net=None,
                    optimizer=None,
                    loss_function=None,
                    save_losses=True,
                    lr=1e-3):
        super(NeuralNet, self).__init__()
        self.key = key
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self._optimizer = optimizer
        self._loss_function = loss_function
        self.activation = self._get_activation_function(activation)
        self.losses = [] if save_losses else None
        self.lr = lr
        assert self.activation is not None, f'Invalid activation function: {activation}'

        self.net = net
        if self.net is None:
            layers = [nn.Linear(input_size, hidden_sizes[0]), self.activation]
            for i in range(len(hidden_sizes) - 1):
                layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
                layers.append(self.activation)
            layers.append(nn.Linear(hidden_sizes[-1], output_size))
            # layers.append(nn.Sigmoid())
            self.net = nn.Sequential(*layers)

    def __str__(self):
        return f'NeuralNet(key={self.key}, layers={[self.input_size] + self.hidden_sizes + [self.output_size]})'

    def _get_activation_function(self, name):
        if name is None:
            return nn.LeakyReLU()
        if isinstance(name, str):
            if name == 'tanh':
                return nn.Tanh()
            if name == 'sigmoid':
                return nn.Sigmoid()
            if name == 'leaky_relu':
                return nn.LeakyReLU()
        if isinstance(name, nn.Module):
            return name
        raise ValueError(f'Invalid activation function: {name}')

    def clone(self, key=None):
        return NeuralNet(
            key or self.key,
            self.input_size,
            self.hidden_sizes,
            self.output_size,
            self.activation,
            net=deepcopy(self.net)
        )
    
    def forward(self, observation, store_gradients=False, activate_output=False):
        """Run the neural net forward on the observation.
        :param observation: the input to run the neural net on
        :param store_gradients: whether to store the gradients for backpropagation
        :param activate_output: whether to apply the activation function to the output
            boolean for running default
            string to specify activation function
            activation function to use
        :return: the output of the neural net"""  
        if isinstance(observation, list):
            observation = torch.tensor(observation)
        if store_gradients:
            return self.net(observation)
        else:
            # if we aren't doing any backpropagation, we don't need to keep track of the intermediate values
            with torch.no_grad():
                return self.net(observation)

    @property
    def key(self):
        if self._key is None:
            self._key = uuid.uuid4().hex
        return self._key

    @key.setter
    def key(self, value):
        self._key = uuid.uuid4().hex if value is None else value

    @property
    def loss_function(self):
        if not self._loss_function:
            self._loss_function = nn.MSELoss()
            # self._loss_function = nn.BCELoss()
        return self._loss_function

    @property
    def optimizer(self):
        if not self._optimizer:
            # torch.optim.SGD(self.parameters(), lr=lr or 0.01)
            self._optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return self._optimizer
            
    def train(self, observation, target, loss_function=None):
        """Train the neural net on the observation and target.
        :param observation: the input to run the neural net on
        :param target: the target output of the neural net
        # :param lr: the learning rate
        :param loss_function: the loss function to use
            - if None, use the default loss function (MSE)
            - nn.CrossEntropyLoss(), nn.BCELoss(), nn.NLLLoss(), etc.
        """
        # if isinstance(observation, list):
        #     observation = torch.tensor(observation)
        loss_function = loss_function or self.loss_function
        prediction = self.forward(observation, store_gradients=True)
        loss = loss_function(prediction, target)
        # zero gradients, perform a backward pass, and update the weights
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if isinstance(self.losses, list):
            self.losses.append(loss.item())
        return loss

    def train_gan(self, discriminator, observation, target, output=None):
        """Train the neural net on the observation and target.
        :param discriminator: the discriminator neural net
        :param observation: the input to run the neural net on
        :param target: the target output of the neural net
        :param output: the output of the neural net
        """
        if output is None:
            output = self.forward(observation, store_gradients=True)
        d_output = discriminator.forward(output, store_gradients=True)
        loss = discriminator.loss_function(d_output, target)
        # zero gradients, perform a backward pass, and update the weights
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if isinstance(self.losses, list):
            self.losses.append(loss.item())
        return loss

    @staticmethod
    def load(filename=None, device=None):
        filename = filename or os.path.join(out_dir, 'best_genome')
        # from out.evolution_best_meta import input_size, hidden_sizes, output_size
        meta = importlib.import_module(f'out.{filename}_meta')
        net = NeuralNet(filename, meta.input_size, meta.hidden_sizes, meta.output_size)
        net.load_state_dict(torch.load(f'{filename}.pt', map_location=device))
        if os.path.exists(os.path.join(out_dir, f'{filename}_losses.py')):
            with open(os.path.join(out_dir, f'{filename}_losses.py'), 'rb') as f:
                net.losses = pickle.load(f)
        return net

    def save(self, filename=None):
        filename = filename or os.path.join(out_dir, 'best_genome')
        torch.save(self.state_dict(), f'{filename}.pt')
        with open(os.path.join(out_dir, f'{filename}_meta.py'), 'w') as f:
            f.write(f'input_size = {self.input_size}\n')
            f.write(f'hidden_sizes = {self.hidden_sizes}\n')
            f.write(f'output_size = {self.output_size}\n')
        if self.losses:
            with open(os.path.join(out_dir, f'{filename}_losses.py'), 'wb') as f:
                pickle.dump(self.losses, f)


if __name__ == "__main__":
    input_size = 10
    hidden_sizes = [20, 30, 40]
    output_size = 5
    
    obs = torch.rand(1, input_size)
    g1 = NeuralNet('g1', input_size, hidden_sizes, output_size)
    # g2 = g1.clone(key='g2')
    print(g1)
    # print(g2)
    # print('------------------')
    print(g1.forward(obs))
    # print(g2.forward(obs))
    # print('------------------')
    # g1.train(obs, torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]]))
    # print(g1.forward(obs))
    # print(g2.forward(obs))

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
