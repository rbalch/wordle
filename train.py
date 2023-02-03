import neat
import os
import utils
from dojo import Dojo
import novelty_archive as archive
import reporters
from wordle import Wordle

LOAD = False
INITIAL_WORD_COUNT = 1
CHECKPOINT = 10
THREADS = 1
GAME_CHECK = 10
ADD_NEW_WORD = 125
config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    utils.config_file)
config.genome_config.add_activation(
    'leaky_relu6', utils.leaky_relu6
)

novel_archive = archive.NoveltyArchive()
wordle = Wordle()
dojo = Dojo()

if LOAD:
    print('loading population...')
    population = neat.Checkpointer.restore_checkpoint('neat-checkpoint')
else:
    print('initializing population...')
    population = neat.Population(config)

population.add_reporter(neat.StdOutReporter(True))
population.add_reporter(reporters.StatsReporter())
population.add_reporter(reporters.GameCheckReporter(
        wordle=wordle,
        generation_interval=GAME_CHECK
    )
)
population.add_reporter(
    neat.Checkpointer(
        generation_interval=CHECKPOINT,
        time_interval_seconds=None,
        filename_prefix=os.path.join(utils.out_dir, 'neat-checkpoint-')
    )
)
generation = 0
wordle.add_new_answer('tares')


def get_fitness(net):
    score = 0.0
    games = {}
    for answer in wordle.answers:
        games[answer] = {
            'guesses': [],
            'scores': [],
        }
        net.reset()
        obs = dojo.reset(answer)
        done = False
        while not done:
            output = net.activate(obs)
            action = wordle.get_word(output)
            obs, reward, done, _ = dojo.step(action)
            score += reward
            games[answer]['guesses'].append(action)
            games[answer]['scores'].append(reward)
        
    return score, games


def eval_genome(genome, config):
        net = neat.nn.RecurrentNetwork.create(genome, config)
        fitness, data = get_fitness(net)
        return fitness


def eval_genomes(genomes, config):
    n_items_map = {}
    fitness_map = {}
    print(novel_archive)

    for genome_id, genome in genomes:
        net = neat.nn.RecurrentNetwork.create(genome, config)
        # net = neat.nn.FeedForwardNetwork.create(genome, config)
        fitness, data = get_fitness(net)
        fitness_map[genome_id] = fitness
        n_item = archive.NoveltyItem(
            generation=generation,
            genomeId=genome_id,
            data=list(set(
                [g for k, v in data.items() for g in v['guesses']]
            )),
        )
        n_items_map[genome_id] = n_item

    for genome_id, genome in genomes:
        novelty = novel_archive.evaluate_novelty_score(
            nitem=n_items_map[genome_id],
            nitems=n_items_map.values(),
        )
        genome.fitness = fitness_map[genome_id] + novelty


utils.clean_output()

if THREADS > 1:
    pe = neat.ThreadedEvaluator(THREADS, eval_genome)
    best_genome = population.run(pe.evaluate, 10)
else:
    while True:
        best_genome = population.run(eval_genomes, ADD_NEW_WORD)
        wordle.add_new_answer()
    # best_genome = population.run(eval_genomes, 1)
