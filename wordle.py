import random
import utils


WORDS = {}
_i = 0
while len(WORDS) < 16384: # this is the max number you can get with 14 length binary
    WORDS[f'{len(WORDS):014b}'] = utils.master_dict[_i]
    _i = _i+1 if _i < len(utils.master_dict)-1 else 0


class Wordle:

    def __init__(self):
        self.answers = []

    def add_new_answer(self, word=None, num=None):
        num = num or 1
        if isinstance(word, list):
            for w in word:
                self.add_new_answer(w)
        elif word:
            print(f'--> adding: {word}')
            self.answers.append(word)
        elif num:
            for _ in range(num):
                if len(self.answers) == 0:
                    new_word = random.choice(list(WORDS.values()))
                else:
                    new_word = self.answers[0]
                    while new_word in self.answers:
                        new_word = random.choice(list(WORDS.values()))
                self.answers.append(new_word)
                print(f'--> adding: {new_word}')

    def get_word(self, output):
        if isinstance(output, str):
            return WORDS[output]
        return WORDS[
            ''.join([str(int(x > 0)) for x in output])
        ]
    
    def get_key(self, word):
        return list(WORDS.keys())[list(WORDS.values()).index(word)]
