import src.ReinforcementLearning.Maze as Maze
import numpy as np
from collections import defaultdict
import math
import matplotlib.pyplot as plt

end_state = (5, 5)
discount_factor = 0.90
maze = Maze.Maze(10, end_pos=end_state, barriers=[(5, 0), (5, 1)])
actions = ['Up', 'Down', 'Right', 'Left']
state_action_value = defaultdict(lambda: 0)
returns = defaultdict(lambda: [])
episode_lens = []
for i in range(1000):

    #Generate Episode
    s_t = (np.random.randint(0, 10), np.random.randint(0, 10))
    episode = []
    while s_t != end_state:
        best_q = -math.inf
        a_t = []

        if np.random.random() > 0.1:
            for a in actions:
                val = state_action_value[(s_t, a)]
                if val > best_q:
                    a_t = [a]
                    best_q = val
                elif val == best_q:
                    a_t.append(a)
            a_t = a_t[np.random.randint(0, len(a_t))]

        else:
            a_t = actions[np.random.randint(0, len(actions))]

        s_t1, reward = maze.take_action(s_t, a_t)
        episode.append([s_t, a_t, reward])
        s_t = s_t1

    #Learn
    G = 0
    for step in range(len(episode)-1, 0, -1):
        s = episode[step][0]
        a = episode[step][1]
        r = episode[step][2]

        G = G*discount_factor + r

        returns[(s, a)] = returns[(s, a)] + [G]
        state_action_value[(s, a)] = np.mean(returns[(s, a)])

    episode_lens.append(len(episode))

plt.plot(episode_lens)
plt.show()
