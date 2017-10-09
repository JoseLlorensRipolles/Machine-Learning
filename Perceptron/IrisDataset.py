from Perceptron import perceptron

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
    print(perceptron(objects, labels))
