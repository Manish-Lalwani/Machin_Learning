
# reference video for calculating simple linear regression https://www.youtube.com/watch?v=JvS2triCgOY
#====to calculate h(xi)= b0 + b1xi====#
#b0 is y-intercept 
#b1 is slope

'''
xbar = mean of x = mean_x
ybar = mean of y = mean_y

b1 = sum(x-xbar).(y-ybar) / sum((x-xbar)square)

therefore for b1 we have to calculate  
(x-xbar), (y-ybar),  sum(x-xbar),  sum((x-xbar)square)
(x-meanx),(y-meany), sum(x-meanx), sum((x-meanx)square)

'''




#import
import numpy as np 


# Dataset
x = np.array([1,2,3,4,5])
y = np.array([2,4,5,4,5])


#calculating using normal method
'''

# mean of x and y vector 
meanx = np.mean(x)
meany = np.mean(y) 


# calculating sum(x-xbar).(y-ybar)
x_min_xbar_into_y_min_ybar = 0
for i in x:
    x_min_xbar_into_y_min_ybar = x_min_xbar_into_y_min_ybar + ( (x[i-1] - meanx) * (y[i-1] - meany) )
    print(x_min_xbar_into_y_min_ybar)
'''



#calculating sum((x-xbar).(y-ybar)) using numpy
sum_of_x_min_xbar_into_y_min_ybar = 0
sum_of_x_min_xbar_into_y_min_ybar = np.sum((x - x.mean())*(y-y.mean()))

#calculating sum((x-xbar)square) using numpy
sum_of_x_min_xbar_square =np.sum(np.square(x-x.mean()))



#calculating b1 formula : b1 = sum( (x-xbar).(y-ybar) ) / sum( (x-xbar)square) )
b1 = sum_of_x_min_xbar_into_y_min_ybar / sum_of_x_min_xbar_square


#calculating b0 :
#   will substitute in formula y = b0 + b1 as from calculating the mean we have got base case values fro x and y
#   which are x = 3 , y = 4 (we got it from calcuating the mean)
#formula    y = b0 + b1.x
# therefore b0 = b1.x - y
b0 = b1*meanx - meany

#Training model
