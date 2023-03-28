import utils, random, torch, collections
from dojo import Dojo
from wordle import Wordle

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    print(f'using cuda: {torch.cuda.get_device_name(0)}')

bot = utils.NeuralNet('test', 651, [651]*500, 14)
# bot.load()
dojo = Dojo(obs_float=True)
wordle = Wordle(answers_maxlen=100)

# wordle.add_new_answer('tares')
wordle.add_new_answer(num=2)
generation = 0
games = collections.deque([0], maxlen=100)
losses = []

while sum(games) / len(games) < 0.5:
    for answer in wordle.answers:
        generation += 1
        obs = dojo.reset(answer)
        done = False
        while not done:
            guess = bot.forward(obs)
            guess = wordle.get_word(guess)
            obs, reward, done, _ = dojo.step(guess)
        print(f'({sum(games) / len(games)}) {dojo}')
        if answer not in dojo.guesses:
            # dojo.guesses.pop()
            # dojo.scores.pop()
            stop = random.randint(1, 4)
            dojo.guesses = dojo.guesses[0:stop]
            dojo.scores = dojo.scores[0:stop]
            key = wordle.get_key(answer)
            target = torch.tensor([
                random.uniform(-1.0, -.1) if int(x) == 0 
                else random.uniform(.1, 1.0) for x in key
            ])
            # print(f'--> answer: {answer} | key: {key} | target: {target}')
            # print(dojo.observation)
            games.append(0)
            loss = bot.train(dojo.observation, target)
            losses.append(loss.item())
            # print(f'----> loss: {loss}')
        elif answer in dojo.guesses and generation % 500 == 0:
            bot.save()
        else:
            games.append(1)
            bot.save()

    with open('out/losses.py', 'w') as out:
        out.write(f'losses={losses}')
    # wordle.add_new_answer(num=1)


# bot.load()
# obs = dojo.reset(answer='tares')
# done = False

# while not done:
#     guess = wordle.get_word(bot.forward(obs))
#     score = dojo.calcScore(guess)
#     print(f'guess: {guess}')
#     print(f'--> score: {score}')
#     if dojo.remainingMoves > 1:
#         obs, reward, done, _ = dojo.step(guess, score=score)
#         guess = wordle.get_word(bot.forward(obs))
#     else:
#         done = True
