import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt


def make_num_valid(matrix, row, column):
    num = matrix[row, column]
    if math.ceil(num) - num < 0.00000000001:
        matrix[row, column] = math.ceil(num)
    elif num - math.floor(num) < 0.00000000001:
        matrix[row, column] = math.floor(num)


'''
@:param matrix
find the pivot positions 
change the rows so that the pivot not be 0
making the ones below pivot zero
'''


def make_zero(matrix, rows, columns):
    pivot = 0
    for j in range(columns + 1):
        for i in range(pivot, rows):
            if matrix[i][j] != 0:
                matrix[[pivot, i]] = matrix[[i, pivot]]
                for k in range(i + 1, rows):
                    if matrix[k][j] != 0:
                        tmp = -1 * matrix[k][j] / matrix[pivot][j]
                        for l in range(columns + 1):
                            matrix[k][l] = matrix[pivot][l] * tmp + matrix[k][l]
                            make_num_valid(matrix, k, l)
                pivot += 1
                break
    return matrix, pivot


'''
@:param matrix
finding the pivots 
making the ones above pivot zero
'''


def make_zero_backward(matrix, rows, columns):
    pivot = 0
    for j in range(columns + 1):
        for i in range(pivot, rows):
            if matrix[i][j] != 0:
                for k in range(0, i):
                    if matrix[k][j] != 0:
                        tmp = -1 * matrix[k][j] / matrix[pivot][j]
                        for l in range(columns + 1):
                            matrix[k][l] = matrix[pivot][l] * tmp + matrix[k][l]
                            make_num_valid(matrix, k, l)
                pivot += 1
                break
    return matrix


'''
converting the pivots to 1 by dividing its row by the pivot
'''


def change_pivots(matrix, rows, columns):
    pivot = 0
    for j in range(columns):
        for i in range(pivot, rows):
            if pivot == rows:
                break
            if matrix[i][j] != 0:
                matrix[i:i + 1] = matrix[i:i + 1] / matrix[i][j]
                pivot += 1
                break
    return matrix


df = pd.read_csv('GOOGL.csv')
Open_list = df.head(-10)['Open']

inputs = np.empty([len(Open_list), 1])
one = np.ones([len(Open_list), 1])
Open_numpy = np.empty([len(Open_list), 1])
for i in range(len(Open_list)):
    inputs[i] = i + 1
    Open_numpy[i] = Open_list[i]
'''
Linear Regression
In this section at first we make Augmented matrix and then we calculate the value of Beta matrix
'''
A = np.hstack((one.reshape(len(Open_list), -1), inputs.reshape(len(Open_list), -1)))
AT = A.transpose()
matrix_ATA = AT.dot(A)
matrix_ATB = AT.dot(Open_numpy)
answer_for_beta = np.hstack((matrix_ATA, matrix_ATB))
answer_for_beta, pivotCount1 = make_zero(answer_for_beta, 2, 2)
answer_for_beta = make_zero_backward(answer_for_beta, 2, 2)
answer_for_beta = change_pivots(answer_for_beta, 2, 2)
B1 = answer_for_beta[0][2]
B2 = answer_for_beta[1][2]


inputs_pow2 = np.empty([len(Open_list), 1])
for i in range(len(Open_list)):
    inputs_pow2[i] = pow(inputs[i], 2)
'''
Non Linear Regression
In this section at first we make Augmented matrix and then we calculate the value of Beta matrix
'''
arr_reg = np.hstack((A.reshape(len(Open_list), -1), inputs_pow2.reshape(len(Open_list), -1)))
arr_reg_transpose = arr_reg.transpose()
matrix_ATA_2 = arr_reg_transpose.dot(arr_reg)
matrix_ATB_2 = arr_reg_transpose.dot(Open_numpy)
answer_reg_for_beta = np.hstack((matrix_ATA_2, matrix_ATB_2))
answer_reg_for_beta, pivotCount1 = make_zero(answer_reg_for_beta, 3, 3)
answer_reg_for_beta = make_zero_backward(answer_reg_for_beta, 3, 3)
answer_reg_for_beta = change_pivots(answer_reg_for_beta, 3, 3)
B_reg1 = answer_reg_for_beta[0][3]
B_reg2 = answer_reg_for_beta[1][3]
B_reg3 = answer_reg_for_beta[2][3]



actual_list = df.tail(10)['Open']
whole_list = df['Open'].to_list()
'''
In this section we print the calculated values made from linear regression and the actual values and errors
'''
print("***Linear Regression*** : ")
for k in range(len(whole_list) - 10, len(whole_list)):
    print("calculated value: ", B2 * k + B1)
    print("actual value: ", actual_list[k])
    print("error: ", B2 * k + B1 - actual_list[k])
    print()

'''
In this section we print the calculated values made from non linear regression and the actual values and errors
'''
print("***Non Linear Regression*** : ")
calculated_numpy = np.empty([len(whole_list), 1])
for m in range(len(whole_list) - 10, len(whole_list)):
    print("calculated value: ", B_reg3 * m * m + B_reg2 * m + B_reg1)
    print("actual value: ", actual_list[m])
    print("error: ", B_reg3 * m * m + B_reg2 * m + B_reg1 - actual_list[m])
    print()

for z in range(len(whole_list)):
    calculated_numpy[z] = B_reg3 * z * z + B_reg2 * z + B_reg1
print(calculated_numpy)
'''
In this section we draw the diagram
'''
plt.plot(calculated_numpy, color='blue', label='calculated polynomial')
plt.scatter(range(len(whole_list)), whole_list, color='red', label='actual value')
plt.legend(loc='upper left')
plt.show()
