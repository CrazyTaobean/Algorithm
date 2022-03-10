"""
    author:Daobin Zhu
    description:achieve forward algorithm
    date:2022/03/10
"""
import numpy as np


class HMMForwardAlgorithm:
    def __init__(self, A, B, pi):
        """
        :param A:Transition probability matrix
        :param B:Observed probability distribution
        :param pi:Initial probability distribution
        """
        self.A = A;
        self.B = B;
        self.pi = pi;

    """
        A = [0.5, 0.2, 0.3],
            [0.3, 0.5, 0.2],
            [0.2, 0.3, 0.5]
        B = [0.5, 0.5],
            [0.4, 0.6],
            [0.7, 0.3]
        pi = [0.2],
             [0.4],
             [0.4]
    """

    def forward(self, A, B, pi, O):
        """
        :param A:Transition probability matrix
        :param B:Observed probability distribution
        :param pi:Initial probability distribution
        :param O:Observed sequence
        :return:
        """
        N, T = np.shape(A)[0], np.shape(O)[0]
        alpha = np.zeros((T, N))
        for t in range(T):
            ## set up initial value
            if t == 0:
                alpha[t] = np.multiply(pi.T, B[:, O[t]])
                print('alpha[0]:\n', alpha[t])
            else:
                for n in range(N):
                    alpha[t][n] = np.multiply(np.sum(np.multiply(alpha[t - 1], np.array(A)[:, n])), B[:, O[t]][n])
                print('alpha[{}]:\n{}'.format(t, alpha[t]))
                print('now alpha is:\n', alpha[0:t+1, :])
        return np.sum(alpha[-1])

    def GetProb(self, O):
        """
        :param O: return observed seq O's occurred probability
        """
        return self.forward(self.A, self.B, self.pi, O)


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
    hmm = HMMForwardAlgorithm(A, B, pi)

    O = [0, 1, 0]  # 0 represent red, 1 represent white，this example mean(red, white, red) observed sequence
    prob = hmm.GetProb(O)
    print('\ncalculated by the forward algorithm of HMM: observed sequence', O, 'occurred prob is:', prob)


if __name__ == '__main__':
    Test()
