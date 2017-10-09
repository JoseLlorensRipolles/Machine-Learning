import numpy as np

def perceptron(data, labels, max_iterations = 5000, alpha = 1.0, beta = 0.1):
    data = np.array(data)
    object_length = len(data[0])
    class_number = len(set(labels))
    weights = []

    for i in range(class_number):
        weights.append([])
        for j in range(object_length):
            weights[i].append(float(0))
    weights = np.array(weights)

    print(weights)
    for i in range(max_iterations):
        err = False
        for j in range(len(data)):
            xn = np.array(data[j])
            cn = labels[j]
            wcn = np.array(weights[cn])
            aux = np.dot(wcn, xn)

            for c in range(class_number):
                if c is not cn:
                    if np.dot(weights[c], xn) + beta > aux:
                        weights[c] -= alpha*xn
                        err = True
            if err:
                weights[cn] += alpha*xn
        if not err:
            break

    well_classified = 0
    for i in range(len(data)):
        object = data[i]
        max = 0
        classified_in = -1
        for c in range(len(weights)):
            res = np.dot(object, weights[c])
            if res > max:
                max = res
                classified_in = c
        if classified_in is labels[i]:
            well_classified += 1

    print('Well classified:',well_classified,'of a total of', len(data), 'objects')




    pass


if __name__ == "__main__":

    raw_data = open('Resources/IrisDataset', 'r').read()
    raw_objects = raw_data.splitlines()
    objects = []

    for i in range(len(raw_objects)):
        objects.append([1])


    raw_labels = []
    labels = []

    for i in range(len(raw_objects)):
        raw_object = raw_objects[i].split(',')

        for j in range(len(raw_object)-1):
            objects[i].append(float(raw_object[j]))
        raw_labels.append(raw_object[4])


    for raw_label in raw_labels:
        if raw_label == 'Iris-setosa':
            labels.append(0)
        if raw_label == 'Iris-versicolor':
            labels.append(1)
        if raw_label == 'Iris-virginica':
            labels.append(2)
    perceptron(objects, labels)
