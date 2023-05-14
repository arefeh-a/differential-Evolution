import random
import numpy as np
import math
from time import time


def first_generate_board(board_size):
    board = []
    for i in range(1, board_size + 1):
        board.append(i)
    random.shuffle(board)
    return board


def fitness(gen):
    horizontal_collisions = sum([gen.count(queen) - 1 for queen in gen]) / 2

    n = len(gen)
    maxFitness = math.comb(n, 2)
    left_diagonal = [0] * 2 * n
    right_diagonal = [0] * 2 * n
    for i in range(n):
        left_diagonal[i + gen[i] - 1] += 1
        right_diagonal[len(gen) - i + gen[i] - 2] += 1

    diagonal_collisions = 0
    for i in range(2 * n - 1):
        counter = 0
        if left_diagonal[i] > 1:
            counter += left_diagonal[i] - 1
        if right_diagonal[i] > 1:
            counter += right_diagonal[i] - 1
        diagonal_collisions += counter / (n - abs(i - n + 1))

    return int(maxFitness - (horizontal_collisions + diagonal_collisions))


def mutation(gen1, gen2, gen3, F):
    n = len(gen1)

    gen1 = np.array(gen1)
    gen2 = np.array(gen2)
    gen3 = np.array(gen3)

    donor_vec = gen3 + F * (gen2 - gen1)
    donor_vec = donor_vec.tolist()
    for row in donor_vec:
        if row < 1:
            donor_vec[donor_vec.index(row)] = 1
        elif row > n:
            donor_vec[donor_vec.index(row)] = n


    return donor_vec


def crossover(gen0, donor_vec, CR):
    n = len(gen0)
    for i in range(n):
        t = random.random()
        if t >= CR:
            donor_vec[i] = gen0[i]

    return donor_vec


def selection(gen0, target_vec):
    if fitness(gen0) > fitness(target_vec):
        new_gen = gen0
    else:
        new_gen = target_vec

    return new_gen


def differentialEvolution(board_size, F, CR, NP):
    # making the first generation
    gens = []
    for i in range(NP):
        gens.append(first_generate_board(board_size))
    #
    maxFitness = math.comb(board_size, 2)
    while True:
        for gen in gens:
            lst = [i for i in gens if i != gen]
            gen1, gen2, gen3 = random.sample(lst, 3)
            donor_vec = mutation(gen1, gen2, gen3, F)
            target_vec = crossover(gen, donor_vec, CR)
            if fitness(target_vec) == maxFitness:
                return target_vec
            gens[gens.index(gen)] = selection(gen, target_vec)


if __name__ == '__main__':
    start = time()
    board_size = 8
    NP = 5*board_size
    print(differentialEvolution(board_size, 1, 0.9, NP))
    end = time()
    print(end - start)
