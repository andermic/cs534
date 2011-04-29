#! /usr/bin/python
import matplotlib.pyplot as plot

#XMIN = -2
#XMAX = 7

#dec_x = [XMIN, XMAX]
#dec_y = [-(w[0] + XMIN*w[1])/w[2], -(w[0] + XMAX*w[1])/w[2]]
dec1_x = [25,25]
dec1_y = [0,25]
dec2_x = [0,40]
dec2_y = [15,15]
dec3_x = [10,10]
dec3_y = [0,15]
dec4_x = [5,5]
dec4_y = [15,25]

#plot.axis('equal')
plot.title(r'Decision Boundaries')
plot.xlabel('$x_1$')
plot.ylabel('$x_2$', rotation='horizontal')
plot.plot(dec1_x, dec1_y, color='black')
plot.plot(dec2_x, dec2_y, color='black')
plot.plot(dec3_x, dec3_y, color='black')
plot.plot(dec4_x, dec4_y, color='black')
plot.show()
