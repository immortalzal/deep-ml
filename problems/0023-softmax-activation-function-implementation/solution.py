import math

def softmax(scores: list[float]) -> list[float]:
    sum = 0
    buff = -1000000000
    for i in range(len(scores)):
        buff = max(scores[i], buff)
    for i in range(len(scores)):
        sum += math.exp(scores[i] - buff)
    for i in range(len(scores)):
        scores[i] = math.exp(scores[i] - buff) / sum
    return scores
    pass