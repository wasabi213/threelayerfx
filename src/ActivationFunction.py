import numpy as np

sigmoid_range = 34.538776394910684

def sigmoid(x):

    #print(x)
    #print(-np.clip(x, -sigmoid_range, sigmoid_range))
    #print(np.exp(-np.clip(x, -sigmoid_range, sigmoid_range)))
    #quit()
    #print(1.0 / (1.0 + np.exp(-np.clip(x, -sigmoid_range, sigmoid_range))))
    #quit()
    return 1.0 / (1.0 + np.exp(-np.clip(x, -sigmoid_range, sigmoid_range)))

    #ここをreluにしてみる。



def derivative_sigmoid(o):
    #print(o)
    #quit()
    return o * (1.0 - o)

def relu(i):

    x = i[0][0]


    if x > 0.0 == True:
        ret = x * 1
    else:
        ret = x * 0
    return ret
