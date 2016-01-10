from seq2seq import Seq2seq
from random import randint

"""
    Toy task 3: counting
"""

def main():
    seq2seq = Seq2seq()

    last_seq = None
    cost = 0

    for i in range(100000):

        X = [randint(1, 2) for _ in range(randint(1, 10))]
        Y = [x for x in X if x == 1]
        cost += seq2seq.train(X, Y)

        if i % 1000 == 0:
            print i, '\t', cost / 1000
            cost = 0

            X = [randint(1, 2) for _ in range(randint(1, 10))]
            Y = seq2seq.predict(X)

            print X, '->', Y

            seq2seq.lr /= 2


if __name__ == "__main__":
    main()
