import pickle
from neat.reporting import BaseReporter


class StatsReporter(BaseReporter):

    def __init__(self, generation_interval=10):
        BaseReporter.__init__(self)
        self.generation = None
        self.best_score = -10000000
        self.best_scores = []
        self.average_scores = []
        self.species_scores = []

    def start_generation(self, generation):
        self.generation = generation

    # def end_generation(self, config, population, species_set):
    #     pass

    def post_evaluate(self, config, population, species, best_genome):
        if best_genome.fitness > self.best_score:
            self.best_score = best_genome.fitness
            self._save_best_genome(best_genome)
            self._save_config(config)
        self.best_scores.append(best_genome.fitness)
        self.average_scores.append(
            sum([g.fitness for g in population.values()]) / len(population.values())
        )

        species_stats = {}
        for sid, s in species.species.items():
            species_stats[sid] = [
                sum([g.fitness for g in s.members.values()]) / len(s.members),
            ]
        self.species_scores.append(species_stats)

        self._save_scores()

    # def post_reproduction(self, config, population, species):
    #     pass

    # def complete_extinction(self):
    #     pass

    # def found_solution(self, config, generation, best):
    #     pass

    # def species_stagnant(self, sid, species):
    #     pass

    # def info(self, msg):
    #     pass

    def _save_best_genome(self, best_genome):
        with open('out/best_genome.pkl', 'wb') as f:
            pickle.dump(best_genome, f)

    def _save_config(self, config):
        with open('out/config.pkl', 'wb') as f:
            pickle.dump(config, f)

    def _save_scores(self):
        with open('out/best_scores.py', 'w') as out:
            out.write(f'best_scores={self.best_scores}\n')
            out.write(f'average_scores={self.average_scores}\n')
            out.write(f'species_scores={self.species_scores}\n')
