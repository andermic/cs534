#! /usr/bin/python
from random import randint
from math import sqrt
import matplotlib.pyplot as plot

def shuffle(data):
    shuffled = []
    while len(data) != 0:
        insert_index = randint(0, len(data)-1)
        shuffled.append(data[insert_index])
        data = data[:insert_index] + data[insert_index+1:]
    return shuffled

def vector_magnitude(v):
    return sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    
class Perceptron:
    def p_train(self, data):
        w = [0.0, 0.0, 0.0]
        N = len(data)
        print N
        eta = 1
        while True:
            delta = [0.0, 0.0, 0.0]
            for i in range(N):
                x = data[i][:3]
                y = data[i][3]
                u = w[0]*x[0] + w[1]*x[1] + w[2]*x[2]
                if y * u <= 0:
                    delta = [delta[j] - y*x[j] for j in range(len(delta))]
            delta = [delta[j] / N for j in range(len(delta))]
            w = [w[j] - eta*delta[j] for j in range(len(delta))]
            if delta == [0.0, 0.0, 0.0]:
                break
        return w
    
    def p_classify(self, x, w):
        boundary = -w[1]/w[2]
        posit_w2 = w[2] > 0
        above_boundary = (x[2] > boundary * x[1] + w[0] + w[0])
        return {True:1.0,False:-1.0}[posit_w2 == above_boundary]

    def __init__(self):
        # Read in .csv file, turn each line into a canonical form vector
        raw_data = open('twogaussian.csv', 'r').readlines()
    
        canon = []
        for i in range(len(raw_data)):
            raw_data[i] = raw_data[i].strip().split(',')
            raw_data[i] = [1, raw_data[i][1], raw_data[i][2], raw_data[i][0]]
            canon.append([float(j) for j in raw_data[i]])
    
        # Train using the perceptron algorithm
        w = self.p_train(canon)
        print w
    
        # Plot the perceptron decision boundary and data points
        dec_slope = -w[1]/w[2]
        dec_yint = w[0]
    
        dec_x = [-2, 7]
        dec_y = [-2 * dec_slope + dec_yint, 7 * dec_slope + dec_yint]
        
        xs_positive = [i[1] for i in canon if i[3]== 1.0]
        ys_positive = [i[2] for i in canon if i[3]== 1.0]
        xs_negative = [i[1] for i in canon if i[3]== -1.0]
        ys_negative = [i[2] for i in canon if i[3]== -1.0]
    
        plot.axis('equal')
        plot.title(r'Points and Decision Boundary for Perceptron')
        plot.xlabel('$x_1$')
        plot.ylabel('$x_2$', rotation='horizontal')
        plot.plot(dec_x, dec_y, label='Decision Boundary', color='black')
        plot.scatter(xs_positive, ys_positive, label='Positive', color='green')
        plot.scatter(xs_negative, ys_negative, label='Negative', color='red')
        plot.legend(loc='upper left') 
        plot.show()
        plot.savefig('perceptron.png')

class Voted_Perceptron:
    def vp_train(self, data):
        pass

    def vp_classify(self, x, w):
        pass

    def __init__(self):
        pass

Perceptron()
