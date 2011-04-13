#! /usr/bin/python
from random import randint
from random import random
from math import sqrt
import matplotlib.pyplot as plot

def sign(a):
    if a > 0:
        return 1.0
    else:
        return -1.0

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
        above_boundary = (x[2] > boundary * x[1] + w[0])
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
    
        # Plot the perceptron decision boundary and data points
        dec_slope = -w[1]/w[2]
        dec_yint = 3
    
        dec_x = [-2, 7]
        dec_y = [-2 * dec_slope + dec_yint, 7 * dec_slope + dec_yint]
        
        xs_positive = [i[1] for i in canon if i[3] == 1.0]
        ys_positive = [i[2] for i in canon if i[3] == 1.0]
        xs_negative = [i[1] for i in canon if i[3] == -1.0]
        ys_negative = [i[2] for i in canon if i[3] == -1.0]
    
        plot.axis('equal')
        plot.title(r'Points and Decision Boundary for Perceptron')
        plot.xlabel('$x_1$')
        plot.ylabel('$x_2$', rotation='horizontal')
        plot.plot(dec_x, dec_y, label='Decision Boundary', color='black')
        plot.scatter(xs_positive, ys_positive, label='Positive', color='green')
        plot.scatter(xs_negative, ys_negative, label='Negative', color='red')
        plot.legend(loc='upper left') 
        plot.show()

class Voted_Perceptron:
    def vp_train(self, data, epoch):
        w = [[0.0, 0.0, 0.0]]
        c = [0]
        n = 0
        progress = []
        for e in range(epoch):
            data = shuffle(data)
            for i in range(len(data)):
                x = data[i][:3]
                y = data[i][3]
                u = w[n][0]*x[0] + w[n][1]*x[1] + w[n][2]*x[2]
                if y * u <= 0:
                    w.append([w[n][0] + y*x[0], w[n][1] + y*x[1], w[n][2] + y*x[2]])
                    c.append(0)
                    n += 1
                else:
                    c[n] = c[n] + 1
            #Track the performance across epoches
            progress.append(0)
            print len(w)
            for d in data:
                if self.vp_classify(d, w, c) != d[3]:
                    progress[-1] += 1
        return w, c, progress

    def vp_classify(self, x, w, c):
        result = 0
        for i in range(len(c)):
            inner_prod = w[i][0]*x[0] + w[i][1]*x[1] + w[i][2]*x[2]
            result += c[i] * sign(inner_prod) 
        return sign(result)

    def __init__(self):
        EPOCH = 100

         # Read in .csv file, turn each line into a canonical form vector
        raw_data = open('iris-twoclass.csv', 'r').readlines()

        canon = []
        for i in range(len(raw_data)):
            raw_data[i] = raw_data[i].strip().split(',')
            raw_data[i] = [1, raw_data[i][1], raw_data[i][2], raw_data[i][0]]
            canon.append([float(j) for j in raw_data[i]])
    
        # Train using the voted perceptron algorithm
        result = self.vp_train(canon, EPOCH)
        w = result[0]
        c = result[1]
        progress = result[2]

        # Plot the progress of the training program as a function of the
        #  number of epoches
        xs = range(1,101)
        ys = progress

        plot.title(r'Progress of Voted Perceptron as a function of training epoches') 
        plot.xlabel('Epoches')
        plot.ylabel('Misclassifications')
        plot.plot(xs, ys)
        plot.show()

        # Visualize the decision boundary by evaluating some random points
        rand_points = []
        for i in range(1000):
            x = [1.0, random() * 7, random() * 4]
            x.append(self.vp_classify(x, w, c))
            rand_points.append(x)

        # Plot the voted perceptron decision boundary using random values
        xs_positive = [i[1] for i in rand_points if i[3] == 1.0]
        ys_positive = [i[2] for i in rand_points if i[3] == 1.0]
        xs_negative = [i[1] for i in rand_points if i[3] == -1.0]
        ys_negative = [i[2] for i in rand_points if i[3] == -1.0]
    
        plot.title(r'Visualized Decision Boundary for Voted Perceptron')
        plot.xlabel('$x_1$')
        plot.ylabel('$x_2$', rotation='horizontal')
        plot.scatter(xs_positive, ys_positive, label='Positive', color='green')
        plot.scatter(xs_negative, ys_negative, label='Negative', color='red')
        plot.legend(loc='upper left') 
        plot.show()


class Voted_Perceptron_Avg:
    def vp_train_avg(self, data, epoch):
        w_avg = [0.0, 0.0, 0.0]
        w = [0.0, 0.0, 0.0]
        c = [0]
        n = 0
        for e in range(epoch):
            data = shuffle(data)
            for i in range(len(data)):
                x = data[i][:3]
                y = data[i][3]
                u = w[0]*x[0] + w[1]*x[1] + w[2]*x[2]
                if y * u <= 0:
                    w_avg[0] += c[n] * w[0]
                    w_avg[1] += c[n] * w[1]
                    w_avg[2] += c[n] * w[2]
                    w[0] += y*x[0]
                    w[1] += y*x[1]
                    w[2] += y*x[2]
                    c.append(0)
                    n += 1
                else:
                    c[n] = c[n] + 1
        return [i/n for i in w_avg], c

    def vp_classify_avg(self, x, w_avg):
        return sign(w_avg[0]*x[0] + w_avg[1]*x[1] + w_avg[2]*x[2])

    def __init__(self):
        EPOCH = 100

         # Read in .csv file, turn each line into a canonical form vector
        raw_data = open('iris-twoclass.csv', 'r').readlines()

        canon = []
        for i in range(len(raw_data)):
            raw_data[i] = raw_data[i].strip().split(',')
            raw_data[i] = [1, raw_data[i][1], raw_data[i][2], raw_data[i][0]]
            canon.append([float(j) for j in raw_data[i]])
    
        # Train using the voted perceptron w_avg algorithm
        result = self.vp_train_avg(canon, EPOCH)
        w_avg = result[0]
        c = result[1]
 
        # Plot the voted perceptron w_avg decision boundary and data points
        dec_slope = -w_avg[1]/w_avg[2]
        dec_yint = 2.5
    
        print w_avg

        dec_x = [0, 7]
        dec_y = [0 * dec_slope + dec_yint, 7 * dec_slope + dec_yint]
        
        xs_positive = [i[1] for i in canon if i[3] == 1.0]
        ys_positive = [i[2] for i in canon if i[3] == 1.0]
        xs_negative = [i[1] for i in canon if i[3] == -1.0]
        ys_negative = [i[2] for i in canon if i[3] == -1.0]
    
        plot.axis('equal')
        t = r'Points and Decision Boundary for Voted Perceptron with $w_{avg}$'
        plot.title(t)
        plot.xlabel('$x_1$')
        plot.ylabel('$x_2$', rotation='horizontal')
        plot.plot(dec_x, dec_y, label='Decision Boundary', color='black')
        plot.scatter(xs_positive, ys_positive, label='Positive', color='green')
        plot.scatter(xs_negative, ys_negative, label='Negative', color='red')
        plot.legend(loc='upper left') 
        plot.show()

Perceptron()
