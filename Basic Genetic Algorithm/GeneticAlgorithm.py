# Example of genetic algorithm used to maximize the function f(x) = -x^2 + 30 x + 30
import random


def calculate_fun_value(number):
    return (-number**2) + 30*number + 30


def descendants(parents):
    cut1 = random.randint(0, 5)
    cut2 = random.randint(0, 5)

    mother = parents[0]
    father = parents[1]

    heir1 = mother[0:cut1] + father[cut1:]
    heir2 = father[0:cut1] + mother[cut1:]
    heir3 = mother[0:cut2] + father[cut2:]
    heir4 = father[0:cut2] + mother[cut2:]

    return [heir1, heir2, heir3, heir4]


def mutate(population_to_mutate):

    mutated_population = []

    for individual_to_mutate in population_to_mutate:
        rand_number = random.random()
        if rand_number < 0.15:
            bit_index = random.randint(0, 4)
            bit_value = individual[bit_index]
            if bit_value == '1':
                individual_to_mutate = individual[0:bit_index] + '0' + individual[bit_index + 1:]
            else:
                individual_to_mutate = individual[0:bit_index] + '1' + individual[bit_index + 1:]

        mutated_population.append(individual_to_mutate)

    return mutated_population


if __name__ == "__main__":

    # We define our initial population (Arbitrarily chosen) as well as initialize some useful variables
    population = {'01101', '11000', '01000', '10011'}
    best_individual = ''
    best_score = 0
    generations_without_improvement = 0

    # When we've been 1000 generations without getting a better individual we can consider we've finished
    while generations_without_improvement < 1000:
        scores = {}
        score_summation = 0
        generations_without_improvement += 1
        
        # For each individual in our population we calculate his score as the result of the function to maximize with
        # that individual as the input.
        for individual in population:
            score = calculate_fun_value(int(individual, 2))
            scores[individual] = score
            score_summation += score

            # If this individual improves our best we store him and reset the number of generations without improvement
            if score > best_score:
                best_score = score
                best_individual = individual
                generations_without_improvement = 0

        # We choose 2 parents from our population. We will assign the probability of being chosen as the score the
        # individual has got divided by the sum of all scores, this way all probabilities of being chosen sum 1.
        chosen_parents = []
        for i in range(0, 2):
            random_number = random.random()
            probability_summation = 0
            for individual in population:
                probability = scores[individual]/score_summation
                probability_summation += probability
                if probability_summation > random_number:
                    chosen_parents.append(individual)
                    break

        # We calculate the descendants of our chosen parents and (maybe)mutate them
        population = descendants(chosen_parents)
        population = mutate(population)

    print('Our best individual was:', int(best_individual, 2), 'with a score of:', best_score)
