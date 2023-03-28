import matplotlib.pyplot as plt
from out.evolution_data import bestFitness, meanFitness

OFFSET = 0

plt.plot(bestFitness[OFFSET:], color='red')
plt.plot(meanFitness[OFFSET:], color='green')
plt.xlabel('Generation')
plt.ylabel('Max / Average Fitness')
plt.title('Max and Average fitness over Generations')
plt.show()
