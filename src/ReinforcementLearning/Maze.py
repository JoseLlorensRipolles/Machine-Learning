import numpy as np
import matplotlib.pyplot as plt


class Maze:

    def __init__(self, maze_size, end_pos=(1, 1), barriers=None):
        self.maze_size = maze_size
        self.end_pos = end_pos
        self.barriers = barriers

        self.mat = np.array(np.ones((self.maze_size, self.maze_size)))
        for barrier in barriers:
            self.mat[barrier] = 0
        self.mat[self.end_pos] = 2

    def take_action(self,s, a):
        x = s[0]
        y = s[1]

        if a == 'Up':
            if y < self.maze_size - 1:
                y += 1
        elif a == 'Down':
            if y > 0:
                y -= 1
        elif a == 'Right':
            if x < self.maze_size -1 :
                x += 1
        else:
            if x > 0:
                x -= 1

        s_t1 = (x,y)
        reward = 1 if s_t1 == self.end_pos else -1

        return s_t1, reward

    def show(self):
        plt.imshow(self.mat, extent=[0, self.maze_size, 0, self.maze_size], )
        plt.gca().set_xticks(np.arange(0, self.maze_size, self.maze_size / 10))
        plt.gca().set_yticks(np.arange(self.maze_size, 0, -self.maze_size / 10))
        plt.ylim((0, self.maze_size))
        plt.xlim((0, self.maze_size))
        plt.show()
