# import neat
# import pickle
import utils
import torch
from dojo import Dojo
from wordle import Wordle


class Player:

    def __init__(self) -> None:
        # self.genome = self._load_genome()
        self.net = self._load_net()
        self.wordle = Wordle()
        self.dojo = Dojo(obs_float=True)

    def __call__(self):
        total_reward = 0
        while not self.dojo.done:
            guess = self.get_guess()
            score = self.get_score(guess)
            reward = self.dojo.calcReward(guess, score)
            self.dojo.record(guess, score)
            total_reward += reward
            print(f'guess: {guess} score: {score} reward: {reward}')
        print('solved :)' if score == [3, 3, 3, 3, 3] else 'failed :(')
        print(f'total reward: {total_reward}')
        print(f'guesses ({len(self.dojo.guesses)}): {self.dojo.guesses}')

    def _load_config(self):
        with open('out/config.pkl', 'rb') as f:
            config = pickle.load(f)
        return config

    @staticmethod
    def _load_net():
        # return utils.NeuralNet.load('evolution_data')
        # net = utils.NeuralNet('test', 651, [651]*5, 14)
        net = utils.NeuralNet.load(filename='evolution_best', 
                    device=torch.device('cpu'))
        return net

    # def _load_genome(self):
    #     with open('out/best_genome.pkl', 'rb') as f:    
    #         genome = neat.nn.RecurrentNetwork.create(
    #             pickle.load(f), self._load_config()
    #         )
    #     return genome

    # def get_guess(self):
    #     output = self.genome.activate(self.dojo.observation)
    #     return self.wordle.get_word(output)

    def get_guess(self):
        output = self.net.forward(self.dojo.observation)
        return self.wordle.get_word(output)

    def get_score(self, guess):
        print(f'--> guess ({len(self.dojo.guesses) + 1}): {guess}')
        result = ''
        while len(result) != 5:
            result = input('3: green | 2: orange | 1: black\n--> ')
        return [int(x) for x in result]


if __name__ == '__main__':
    Player()()
