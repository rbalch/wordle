import torch
import torch.nn as nn
import __init__ as utils
from neat.genome import DefaultGenomeConfig


class PytorchGenome:

    def __init__(self, key):
        self.key = key
        self.fitness = None
        # self.activation_defs = []

    def __str__(self):
        return f'PytorchGenome(key={self.key}, fitness={self.fitness})'
    
    # def add_activation(self, name, func):
    #     self.activation_defs.add(name, func)

    def configure_crossover(self, genome1, genome2, config):
        """turn this into a crossover of two genomes"""
        pass

    def configure_new(self, config):
        """turn this into a random new genome"""
        pass

    def distance(self, other, config):
        """return the distance between this genome and another - for species"""
        pass

    def mutate(self, config):
        pass
    
    @classmethod
    def parse_config(cls, param_dict):
        return DefaultGenomeConfig(param_dict)

    def size(self):
        """
        Returns genome 'complexity', taken to be
        (number of nodes, number of enabled connections)
        """
        return (0, 0)

    @classmethod
    def write_config(cls, f, config):
        config.save(f)


if __name__ == "__main__":
    # import neat

    # config = neat.Config(
    #     PytorchGenome,
    #     neat.DefaultReproduction,
    #     neat.DefaultSpeciesSet,
    #     neat.DefaultStagnation,
    #     utils.config_file)
    # config.genome_config.add_activation(
    #     'leaky_relu6', utils.leaky_relu6
    # )
    # population = neat.Population(config)

    g = PytorchGenome(0)
    print(g)
