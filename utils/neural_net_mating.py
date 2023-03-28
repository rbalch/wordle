import operator
import random
import torch


class NeuralNetMating:

    def __init__(self, crossover_rate=0.9, mutation_rate=0.05, indpb=0.01, beta=None, population=None):
        """
        :param crossover_rate: likelihood of direct copy of parents
         (no crossover; ie. 0.9 means 90% chance of mating)
        :param mutation_rate: probability an individual will mutate
        :param indpb: probability of each attribute to be mutated if an individual is 
         selected for mutation
        :param beta: how like the parents the child will be (0.8 - 1.2 considered good)
         if this is not set will random sample from that range
            beta == 1 - offspring are dupes of parents
            beta > 1 - offspring farther apart
            beta < 1 - offspring closer
        """
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.indpb = indpb
        self._beta = beta
        self.population = population or []

    def add_layer(self, net, position=None, new_size=None, activation=None):
        """
        updates the model in place; also returns it
        :param net: the neural net to add a layer to
        :param position: the position to add the layer at; or random
        :param new_size: the size of the new layer; or random
        :param activation: the activation function to use; or the net default
        :return: the neural net with the new layer added"""
        position = random.randint(0, len(net.hidden_sizes)-1) if position is None else position
        avg_size = sum(net.hidden_sizes) // len(net.hidden_sizes)
        if new_size is None:
            new_size = random.randint(
                avg_size // 2,
                avg_size * 2
            )
            new_size = 1 if new_size < 1 else new_size
        activation = activation or net.activation

        net.hidden_sizes.insert(position, new_size)
        layers = list(net.net)
        position = 2*(position+1)
        in_size = net.net[position-2].out_features
        out_size = net.net[position].in_features
        layers.insert(position, activation)
        layers.insert(position, torch.nn.Linear(new_size, out_size))
        layers.insert(position, activation)
        layers.insert(position, torch.nn.Linear(in_size, new_size))
        net.net = torch.nn.Sequential(*layers)
        return net

    @property
    def beta(self):
        if self._beta is None:
            return random.uniform(0.8, 1.2)
        return self._beta
    
    def crossover(self, parent1, parent2):
        """
        :param parent1: first parent
        :param parent2: second parent
        :return: 2 children
        with simulated binary crossover you always generate 2 children
        """
        # iterate over the bigger parent
        p1 = max([parent1, parent2], key=lambda _: len(_.hidden_sizes))
        p2 = parent1 if p1 is parent2 else parent2
        child1 = p1#.clone()
        child2 = p2#.clone()
        with torch.no_grad():
            for i, p1_layer in enumerate(p1.net.children()):
                try:
                    p2_layer = p2.net[i]
                    # check both parents have valid weights and biases
                    p1_valid_layer = all([
                        getattr(p1_layer, 'weight', None) is not None,
                        getattr(p1_layer, 'bias', None) is not None,
                    ])
                    p2_valid_layer = all([
                        getattr(p2_layer, 'weight', None) is not None,
                        getattr(p2_layer, 'bias', None) is not None,
                    ])
                    if p1_valid_layer and p2_valid_layer:
                        # bias
                        x = max(p1_layer, p2_layer, key=lambda l: l.bias.size(0))
                        y = p1_layer if x is p2_layer else p2_layer
                        for j, bias1 in enumerate(x.bias):
                            try:
                                child1.net[i].bias[j], child2.net[i].bias[j] = \
                                    self.simulated_binary_crossover(bias1, y.bias[j])
                            except IndexError:
                                break
                        # weight
                        for j, weight1 in enumerate(x.weight):
                            try:
                                for k, value in enumerate(weight1):
                                    child1.net[i].weight[j][k], child2.net[i].weight[j][k] = \
                                        self.simulated_binary_crossover(value, y.weight[j][k])
                            except IndexError:
                                break
                except IndexError:
                    break
            # return [child1, child2]
        
    def delete_layer(self, net, position=None):
        """
        TODO: this reshapes the previous layer to match the next layer
        updates the model in place; also returns it
        :param net: the neural net to delete a layer from
        :param position: the position to delete the layer at; or random
        :return: the neural net with the layer deleted"""
        position = random.randint(0, len(net.hidden_sizes)-1) if position is None else position
        net.hidden_sizes.pop(position)
        layers = list(net.net)
        position = 2*(position+1)
        layers.pop(position)
        layers.pop(position)
        # reshape the layers
        out_shape = layers[position].in_features
        in_shape = layers[position-2].out_features
        if in_shape != out_shape:
            old_layer = layers.pop(position-2)
            layers.insert(position-2, torch.nn.Linear(old_layer.in_features, out_shape))
        net.net = torch.nn.Sequential(*layers)
        return net
    
    def mate(self, parent1, parent2):
        """
        :param parent1: first parent
        :param parent2: second parent
        :return: 2 children
        with simulated binary crossover you always generate 2 children
        """
        if random.random() < self.crossover_rate:
            # parent1, parent2 = self.crossover(parent1, parent2)
            self.crossover(parent1, parent2)
            # parent1 = self.muate(child1)
            # parent2 = self.mutate(child2)
        # else:
        #     children = [parent1, parent2]
        # return [self.mutate(child) for child in children]

    def mutate(self, net):
        # pass through each layer and mutate the weights and biases
        with torch.no_grad():
            for layer in net.net.children():
                # bias
                if hasattr(layer, 'bias'):
                    for j, bias in enumerate(layer.bias):
                        if random.random() < self.indpb:
                            layer.bias[j] = self.mutate_value(bias)
                # weight
                if hasattr(layer, 'weight'):
                    for j, weight in enumerate(layer.weight):
                        for k, value in enumerate(weight):
                            if random.random() < self.indpb:
                                layer.weight[j][k] = self.mutate_value(value)
        # determine if we should add or delete a layer
        if random.random() < self.indpb:
            if random.random() < .5 and len(net.hidden_sizes) > 1:
                self.delete_layer(net)
                
            else:
                self.add_layer(net)

    def mutate_value(self, value, max_drift=.1):
        return random.uniform(
            value - (value * max_drift), 
            value + (value * max_drift)
        )
    
    def simulated_binary_crossover(self, v1, v2, beta=None):
        """
        result is 2 values that are always cumulatively
        the same as the parent:
        beta == 1 - offspring are dupes of parents
        beta > 1 - offspring farther apart
        beta < 1 - offspring closer
        """
        beta = beta or self.beta
        c1 = .5 * ((1+beta)*v1 + (1-beta)*v2)
        c2 = .5 * ((1-beta)*v1 + (1+beta)*v2)
        return c1, c2
    
    def tournament_select_parents(self, tournament_size=3):
        return [
            max([random.choice(self.population) for _ in range(tournament_size)],
                key=operator.attrgetter('fitness'))
            for _ in range(len(self.population))
        ]


if __name__ == '__main__':
    from neural_net import NeuralNet

    # # test mutate
    # net = NeuralNet(key='test', input_size=2, hidden_sizes=[3], output_size=2)
    # mating = NeuralNetMating(crossover_rate=1.0, mutation_rate=1.0, indpb=1.0)
    # print(f'before: {net}')
    # for layer in net.net.children():
    #     if hasattr(layer, 'bias'):
    #         print(layer.bias)
    # print('------------')
    # mating.mutate(net)
    # print(f'after: {net}')
    # for layer in net.net.children():
    #     if hasattr(layer, 'bias'):
    #         print(layer.bias)    

    # test mating
    mating = NeuralNetMating(crossover_rate=1.0)
    nets = [
        NeuralNet(key=_, input_size=2, hidden_sizes=[3], output_size=2)
        for _ in range(2)
    ]

    with torch.no_grad():
        for i, net in enumerate(nets):
            for j, bias in enumerate(net.net[0].bias):
                nets[i].net[0].bias[j] = float(i) + 1.0
            for k, weight in enumerate(net.net[0].weight):
                for l, value in enumerate(weight):
                    nets[i].net[0].weight[k][l] = float(i) + 1.0

    print('parents:')
    for net in nets:
        print('')
        print(f'--> {net}')
        print(f'----> {net.net[0].bias}')
        print(f'----> {net.net[0].weight}')

    print('------------------')
    # children = mating.crossover(*nets)
    mating.mate(*nets)

    print('children:')
    for net in nets:
        print('')
        print(f'--> {net}')
        print(f'----> {net.net[0].bias}')
        print(f'----> {net.net[0].weight}')
    # /test mating

    # # test delete layer
    # mating = NeuralNetMating()
    # net = NeuralNet(key=0, input_size=2, hidden_sizes=[3,4,5], output_size=2)
    # obs = torch.rand(1, net.input_size)
    # print(net)
    # print(net.net)
    # print('------------------')

    # mating.delete_layer(net, position=0)
    # print(net)
    # print(net.net)

    # print('------------------')
    # print(net.forward(obs))
    # # /test delete layer
