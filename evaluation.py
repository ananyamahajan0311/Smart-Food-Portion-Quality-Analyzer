def calculate_accuracy(actual, predicted):
    correct = 0
    for a, p in zip(actual, predicted):
        if a == p:
            correct += 1

    accuracy = (correct / len(actual)) * 100
    return accuracy
