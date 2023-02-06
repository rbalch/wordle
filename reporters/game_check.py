import neat, sys, os
import utils, dojo
from neat.reporting import BaseReporter
from operator import attrgetter


class GameCheckReporter(BaseReporter):

    def __init__(self, wordle, generation_interval=10):
        self.wordle = wordle
        self.dojo = dojo.Dojo()
        self.generation_interval = generation_interval
        self.current_generation = None
        self.best_generation = 0
        self.best_score = -10000000

    def start_generation(self, generation):
        self.current_generation = generation

    def post_evaluate(self, config, population, species, best_genome):
        if self.current_generation % self.generation_interval == 0:
            total = 0
            rows = ['-------| Game check |-------']
            net = neat.nn.RecurrentNetwork.create(best_genome, config)
            # net = neat.nn.FeedForwardNetwork.create(best_genome, config)
            with open(os.path.join(utils.out_dir, 'game_check.txt'), 'w') as f:
                rows.append(f'answers: {self.wordle.answers}')
                games = {}
                for answer in self.wordle.answers:
                    games[answer] = {
                        'guesses': [],
                        'scores': [],
                    }
                    net.reset()
                    obs = self.dojo.reset(answer)
                    done = False
                    while not done:
                        output = net.activate(obs)
                        action = self.wordle.get_word(output)
                        obs, reward, done, _ = self.dojo.step(action)
                        games[answer]['guesses'].append(action)
                        games[answer]['scores'].append(reward)
                    
                    total = sum(games[answer]["scores"])
                    if total > self.best_score:
                        self.best_score = total
                        self.best_generation = self.current_generation

                    rows.append(f'--> answer: {answer}')
                    rows.append(f'guesses: {games[answer]["guesses"]}')
                    rows.append(f'scores: {games[answer]["scores"]} ({sum(games[answer]["scores"])})')
                rows.append(f'-score: {self.best_score} ({total})')
                rows.append(f'-generation: {self.best_generation} ({self.current_generation})')
                f.writelines('\n'.join(rows))
                f.write('\n-----------------------------\n')
