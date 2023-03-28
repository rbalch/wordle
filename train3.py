import utils, random, torch, collections, time
from dojo import Dojo
from wordle import Wordle
from deap import base, creator, tools
from novelty_archive import NoveltyArchive

POPULATION_SIZE = 200
P_CROSSOVER = 0.9
P_MUTATION = 0.1 # likelyhood of an individual to mutate
HALL_OF_FAME_SIZE = 3
ETA = 10.0 # larger numbers offspring are more like parents (10-20 is normal)
BETA = 0.8
SAVE_EVERY = 10 # num of generations to save at
MAX_WORDS = 500 # maximum allowed number of words per generation
NEW_WORD_EVERY = 25 # num of generations to add a new word
NUMBER_OF_NEW_WORDS = 1 # number of new words to add
USE_PREVIOUS_BEST = False # load best individual from file
USE_CUDA = False # use cuda if available and we are using it
INPUT_SIZE=651
HIDDEN_SIZES=[651]*20
OUTPUT_SIZE=14
INDPB = 0.01 # likelyhood of a single gene to mutate
ANIMATION = '|/-\\'

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    print(f'using cuda: {torch.cuda.get_device_name(0)}')
    USE_CUDA = True

# bot = utils.NeuralNet('bot', 651, [651], 14)
dojo = Dojo(obs_float=True)
wordle = Wordle(answers_maxlen=MAX_WORDS)
mating = utils.NeuralNetMating(
    crossover_rate=P_CROSSOVER,
    mutation_rate=P_MUTATION,
    beta=BETA,
)
novelty_archive = NoveltyArchive()


wordle.add_new_answer('tares')
# wordle.add_new_answer(num=9)


def flatten(net):
    output = []
    for layer in net.children():
        valid_layer = all([
            getattr(layer, 'weight', None) is not None,
            getattr(layer, 'bias', None) is not None,
        ])
        if valid_layer:
            output += layer.weight.flatten().tolist()
            output += layer.bias.flatten().tolist()
    return output


def restore(net, weights):
    for layer in net.children():
        valid_layer = all([
            getattr(layer, 'weight', None) is not None,
            getattr(layer, 'bias', None) is not None,
        ])
        if valid_layer:
            layer.weight.data = torch.tensor(weights[:layer.weight.numel()]).reshape(layer.weight.shape)
            weights = weights[layer.weight.numel():]
            layer.bias.data = torch.tensor(weights[:layer.bias.numel()]).reshape(layer.bias.shape)
            weights = weights[layer.bias.numel():]
    return net


def load():
    print('--> loading previous best...')
    net = utils.NeuralNet.load(filename='evolution_best')
    individual = creator.Individual(
        key='evolution_best',
        input_size=INPUT_SIZE,
        hidden_sizes=HIDDEN_SIZES,
        output_size=OUTPUT_SIZE,
    )
    individual.net = net.net
    return individual


def save(generation, bestScores, meanScores, best):
    filename = 'out/evolution_data.py'
    print(f'--> saving: {filename}')
    print(f'----> generation: {generation} | best: {bestScores[-1]} | mean: {meanScores[-1]}')
    with open(filename, 'w') as out:
        out.write(f'generations={generation}\n')
        out.write(f'bestFitness={bestScores}\n')
        out.write(f'meanFitness={meanScores}\n')
    best.save(filename='evolution_best')


def score(individual):
    scores = []
    for answer in wordle.answers:
        obs = dojo.reset(answer)
        done = False
        while not done:
            guess = individual.forward(obs)
            guess = wordle.get_word(guess)
            individual.data.add(guess)
            obs, reward, done, _ = dojo.step(guess)
            scores.append(reward)
    return sum(scores)


def play_game(individual, name, answer=None):
        answer = answer or random.choice(wordle.answers)
        obs = dojo.reset(answer)
        done = False
        rewards = []
        while not done:
            guess = individual.forward(obs)
            guess = wordle.get_word(guess)
            obs, reward, done, _ = dojo.step(guess)
            rewards.append(reward)
        print(f'---- {name} game ----')
        print(individual)
        print(f'---{answer}')
        for guess in dojo.guesses:
            print(f'   - {guess}')
        print(f'--- reward: {sum(rewards)}')
        print(f'fitness: {individual.fitness.values[0]} | novelty: {individual.fitness.values[1]}')
        print('--------------')


def evaluate_novelty(population):
    print('--> evaluating novelty...', end='\r')
    n_items_map = {}

    for individual in population:
        n_item = novelty_archive.create_item(
            genomeId=individual.key,
            data=individual.data,
        )
        n_items_map[individual.key] = n_item
        novelty = novelty_archive.evaluate_novelty_score(
            nitem=n_items_map[individual.key],
            nitems=n_items_map.values(),
        )
        individual.fitness.values = [individual.fitness.values[0], novelty]
    print('--> evaluating novelty... [done]')

toolbox = base.Toolbox()
# creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0))
creator.create("Individual", utils.NeuralNet,
                    fitness=creator.FitnessMax,
                    data=set,
                    novelty=float,
                )
toolbox.register("individualCreator",
                    creator.Individual,
                        key=None,
                        input_size=INPUT_SIZE,
                        hidden_sizes=HIDDEN_SIZES,
                        output_size=OUTPUT_SIZE,
                )
toolbox.register("populationCreator",
                 tools.initRepeat,
                 list,
                 toolbox.individualCreator)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", score)
# toolbox.register("mate", mating.mate)
toolbox.register("mate", tools.cxSimulatedBinary, eta=ETA)
# toolbox.register("mutate", mating.mutate)
toolbox.register("mutate", 
                 tools.mutPolynomialBounded, 
                 low=-6.0, up=6.0, 
                 eta=ETA, 
                 indpb=INDPB
                 )

def main():
    generation = 1
    maxFitnessValues = []
    avgFitnessValues = []
    print(f'Creating population of {POPULATION_SIZE}...')
    population = toolbox.populationCreator(n=POPULATION_SIZE)
    print(f'Hall of fame size: {HALL_OF_FAME_SIZE}')
    hall_of_fame = tools.HallOfFame(HALL_OF_FAME_SIZE)

    if USE_PREVIOUS_BEST: population[0] = load()

    while True:
        print(f'\n{novelty_archive}')
        start = time.time()
        
        # get new fitness values
        fitnessValues = [
            toolbox.evaluate(individual) 
            for individual in population
        ]
        for individual, fitnessValue in zip(population, fitnessValues):
            # individual.fitness.values = [fitnessValue]
            individual.fitness.values = [fitnessValue, 0.0]

        # apply novelty search
        evaluate_novelty(population)

        # record stats
        fitnessValues = [
            individual.fitness.values[0] for individual in population
        ]
        maxFitnessValues.append(max(fitnessValues))
        avgFitnessValues.append(sum(fitnessValues) / len(fitnessValues))

        # update hall of fame
        hall_of_fame.update(population)
        hof_size = len(hall_of_fame.items) if hall_of_fame.items else 0
        best = hall_of_fame.items[0]

        # play a game with the best
        answer = random.choice(wordle.answers)
        play_game(best, 'best', answer=answer)
        # play game with a rando
        play_game(random.choice(population), 'random', answer=answer)

        # move back to cpu
        if USE_CUDA:
            idx = 0
            for individual in population:
                print(f'--> moving to cpu... {ANIMATION[idx % len(ANIMATION)]}', end='\r')
                idx += 1
                individual.to(torch.device('cpu'))
            print('--> moving to cpu... [done]')

        # create children
        print('--> selecting parents...', end='\r')
        offspring = toolbox.select(population, len(population) - hof_size)
        offspring = list(map(toolbox.clone, offspring))
        print('--> selecting parents... [done]')

        idx = 1.0
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            print(f'--> mating... {round((idx/POPULATION_SIZE)*100, 2)}%', end='\r')
            if random.random() < P_CROSSOVER:
                idx += 1
                w1 = flatten(child1.net)
                w2 = flatten(child2.net)
                toolbox.mate(w1, w2)
                restore(child1.net, w1)
                restore(child2.net, w2)
                del child1.fitness.values
                del child2.fitness.values
                child1.key = None
                child2.key = None
                child1.novelty = 0.0
                child2.novelty = 0.0
                child1.data = set()
                child2.data = set()
        print('--> mating... [done]')

        idx = 0
        for mutant in offspring:
            print(f'--> mutating... {ANIMATION[idx % len(ANIMATION)]}', end='\r')
            idx += 1
            if random.random() < P_MUTATION:
                m_weights = flatten(mutant.net)
                toolbox.mutate(m_weights)
                restore(mutant.net, m_weights)
                del mutant.fitness.values
                mutant.key = None
                mutant.novelty = 0.0
                mutant.data = set()
        print('--> mutating... [done]')

        offspring.extend(hall_of_fame.items)
        
        # # reset individuals for next round
        # for individual in offspring:
        #     del individual.fitness.values
        #     individual.novelty = 0.0
        #     individual.data = set()
        
        population[:] = offspring
        end = time.time()

        # move back to cuda
        if USE_CUDA:
            idx = 0
            for individual in population:
                print(f'--> moving back to cuda... {ANIMATION[idx % len(ANIMATION)]}', end='\r')
                idx += 1
                individual.to(torch.device('cuda'))
            print('--> moving back to cuda... [done]')

        if generation % NEW_WORD_EVERY == 0:
            wordle.add_new_answer(num=NUMBER_OF_NEW_WORDS)

        # save data
        print(f'----> generation: {generation} | best: {maxFitnessValues[-1]} | mean: {avgFitnessValues[-1]} | time: {round(end - start, 2)}')
        if generation % SAVE_EVERY == 0:
            save(generation, maxFitnessValues, avgFitnessValues, best)
            # break
        generation += 1


if __name__ == '__main__':
    main()
