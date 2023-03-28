import torch
import utils
import matplotlib.pyplot as plt

device = torch.device('cpu')
bot = utils.NeuralNet('bot', 651, [651]*10, 14)
disc = utils.NeuralNet('discriminator', 14, [14], 1)
bot.load('train3_bot', device=device)
disc.load('train3_disc', device=device)
OFFSET = 0

plt.plot(bot.losses[OFFSET:], color='red')
plt.plot(disc.losses[OFFSET:], color='green')
plt.xlabel('Round')
plt.ylabel('Loss')
plt.title('Losses by Round')
plt.show()
