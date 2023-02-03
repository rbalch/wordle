import matplotlib.pyplot as plt
from out.best_scores import best_scores, average_scores, species_scores

OFFSET = 0

plt.plot(best_scores[OFFSET:], color='red')
plt.plot(average_scores[OFFSET:], color='green')
plt.xlabel('Generation')
plt.ylabel('Max / Average Fitness')
plt.title('Max and Average fitness over Generations')
plt.show()
