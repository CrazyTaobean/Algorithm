"""
    author:Daobin Zhu
    description:achieve backward algorithm
    date:2022/03/10
"""
import numpy as np


class HMMBackwardAlgorithm:
    def __init__(self, A, B, pi):
        self.A = A
        self.B = B
        self.pi = pi

    def backward(self, A, B, pi, O):
        N, T = np.shape(A)[0], np.shape(pi)[0]
        beta = np.zeros((4, N))
        for t in range(T, 0, -1):
            if t == T:
                beta[t] = 1
                print('beta[{}] : {}'.format(t, beta[t]))
            else:
                for n in range(T):
                    beta[t][n] = np.sum(np.multiply(np.multiply(A[n], B[:, O[t]]), beta[t + 1]))
                print('beta[{}] : {}'.format(t, beta[t]))
        beta[0] = np.multiply(np.multiply(pi.T, B[:, O[0]]), beta[1])
        print('beta[0] : {}'.format(beta[0]))
        return np.sum(beta[0])

    def GetProb(self, O):
        """
        :param O: return observed seq O's occurred probability
        """
        return self.backward(self.A, self.B, self.pi, O)


def Test():
    A = np.array([[0.5, 0.2, 0.3],
                  [0.3, 0.5, 0.2],
                  [0.2, 0.3, 0.5]])
    B = np.array([[0.5, 0.5],
                  [0.4, 0.6],
                  [0.7, 0.3]])
    pi = np.array([[0.2],
                   [0.4],
                   [0.4]])
    hmm = HMMBackwardAlgorithm(A, B, pi)

    O = [0, 1, 0]  # 0 represent red, 1 represent white，this example mean(red, white, red) observed sequence
    prob = hmm.GetProb(O)
    print('\ncalculated by the forward algorithm of HMM: observed sequence', O, 'occurred prob is:', prob)


if __name__ == '__main__':
    Test()
