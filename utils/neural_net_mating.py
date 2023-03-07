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
        child1 = p1.clone()
        child2 = p2.clone()
        with torch.no_grad():
            for i, p1_layer in enumerate(p1.net.children()):
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
                    # x = max(p1_layer, p2_layer, key=lambda l: l.weight.size(0))
                    # y = p1_layer if x is p2_layer else p2_layer
                    for j, weight1 in enumerate(x.weight):
                        try:
                            for k, value in enumerate(weight1):
                                child1.net[i].weight[j][k], child2.net[i].weight[j][k] = \
                                    self.simulated_binary_crossover(value, y.weight[j][k])
                        except IndexError:
                            break
            return [child1, child2]
    
    def mate(self, parent1, parent2):
        """
        :param parent1: first parent
        :param parent2: second parent
        :return: 2 children
        with simulated binary crossover you always generate 2 children
        """
        if random.random() < self.crossover_rate:
            children = self.crossover(parent1, parent2)
        else:
            children = [parent1, parent2]
        return [self.mutate(child) for child in children]

    def mutate(self, net, max_drift=.2):
        # if random.random() < self.mutation_rate:
        #     return self.mutate_individual(net)
        return net
    
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
        print(f'--> {net}')
        print(f'----> {net.net[0].bias}')
        print(f'----> {net.net[0].weight}')

    print('------------------')
    children = mating.crossover(*nets)

    print('children:')
    for net in children:
        print(f'--> {net}')
        print(f'----> {net.net[0].bias}')
        print(f'----> {net.net[0].weight}')

    # children = mating.mate(nets[0], nets[1])
    # for child in children:
    #     print('------------------')
    #     for layer in child.hidden_layers:
    #         print(f'layer.weight: {layer.weight}')
    #         print(f'layer.bias: {layer.bias}')
