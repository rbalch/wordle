import math
LETTERS = 'abcdefghijklmnopqrstuvwxyz'


class Dojo:

    def __init__(self):
        self.reset()

    def calcReward(self, guess, score):
        if (self.answer is not None and guess == self.answer) or score == [3, 3, 3, 3, 3] :
            reward = 0.1
            for _ in range(self.remainingMoves):
                reward *= 0.1
        else:
            reward = 1
            # greens
            reward -= .2 * score.count(3)
            # oranges
            reward -= .1 * score.count(2)
            # blacks
            # reward += .1 * score.count(1)
            # dupe guesses
            reward += self.guesses.count(guess) * 1000
            # missing known good letters
            for letter in self.search:
                if letter not in guess:
                    reward += .2
            # analyze letters in guess
            for i, letter in enumerate(guess):
                if letter in self.missing:
                    reward += .2
                elif self.positions[i] and letter not in self.positions[i]:
                    reward += .2
        
        return math.log(1/reward)

    def calcScore(self, guess):
        # 3 green, 2 orange, 1 black
        assert self.answer is not None, "answer must be set before calling calcScore"
        score = []
        letters = [l for l in self.answer]
        for i, l in enumerate(guess):
            if l not in self.answer:
                score.append(1)
            elif l == self.answer[i]:
                score.append(3)
                if l in letters: letters.remove(l)
            else:
                if l in letters:
                    score.append(2)
                    letters.remove(l)
                else:
                    score.append(1)
        return score

    @property
    def done(self):
        return self.answer in self.guesses or self.remainingMoves == 0

    @property
    def observation(self):
        output = []
        for guess, score in zip(self.guesses, self.scores):
            for letter, score in zip(guess, score):
                for _ in LETTERS:
                    if _ == letter:
                        output.append(score)
                    else:
                        output.append(0)
        output += [0 for _ in range((26*5*5) - len(output))]
        output.append(self.remainingMoves)
        return output

    def record(self, guess, score):
        self.guesses.append(guess)
        self.scores.append(score)
        for i, l in enumerate(guess):
            if score[i] == 1:
                self.missing.add(l)
            elif score[i] == 3:
                self.positions[i].add(l)
                self.search.add(l)
            elif score[i] == 2:
                self.not_positions[i].add(l)
                self.search.add(l)

    @property
    def remainingMoves(self):
        return 6 - len(self.guesses)

    def reset(self, answer=None):
        self.answer = answer
        self.guesses = []
        self.scores = []
        self.search = set()
        self.missing = set()
        self.positions = [set() for i in range(5)]
        self.not_positions = [set() for i in range(5)]
        return self.observation

    def step(self, guess, score=None):
        score = score or self.calcScore(guess)
        reward = self.calcReward(guess, score)
        self.record(guess, score)
        info = {'score': score}
        return self.observation, reward, self.done, info


if __name__ == '__main__':
    dojo = Dojo()

    answers = ['tares']
    # guesses = ['daddy', 'filly', 'waurs', 'hecks', 'geoid', 'geyer'] # 1.8
    # guesses = ['tares'] # 16.11
    guesses = ['terms', 'tares'] # 15.0
    # guesses = ['terms', 'terfs', 'tares'] # 13.9
    # guesses = ['terms', 'terms', 'tares'] # 5.8
    # guesses = ['terms', 'terfs', 'teres', 'terek', 'antae', 'teras'] # 5.3
    # guesses = ['terms', 'terfs', 'teres', 'terek', 'antae', 'tares'] # 9.0
    # guesses = ['terms', 'antae', 'teres', 'terek', 'tares'] # 8.8
    # guesses = ['terms', 'teres']

    # guesses = ['weils', 'wekas', 'tapes', 'tapus', 'tanty', 'tares']
    # guesses = ['thelf', 'thine', 'toles', 'tolus', 'tolan', 'tomes']

    for answer in answers:
        score = 0.0
        dojo.reset(answer)
        print(f'answer: {answer}')
        for action in guesses:
            obs, reward, done, _ = dojo.step(action)
            score += reward
            print(f'--> guess: {action} ({reward})')
            if done: break
        print(f'----> score: {score}')