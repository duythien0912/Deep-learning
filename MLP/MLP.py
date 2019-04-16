import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse

N = 100 # number of points per class
d0 = 2 # dimensionality
C = 3 # number of classes
X = np.zeros((d0, N*C)) # data matrix (each row = single example)
y = np.zeros(N*C, dtype='uint8') # class labels

for j in range(C):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
  X[:,ix] = np.c_[r*np.sin(t), r*np.cos(t)].T
  y[ix] = j

# lets visualize the data:
plt.plot(X[0, :N], X[1, :N], 'bs', markersize = 7);
plt.plot(X[0, N:2*N], X[1, N:2*N], 'ro', markersize = 7);
plt.plot(X[0, 2*N:], X[1, 2*N:], 'g^', markersize = 7);

plt.xlim([-1.5, 1.5])
plt.ylim([-1.5, 1.5])
cur_axes = plt.gca()
cur_axes.axes.get_xaxis().set_ticks([])
cur_axes.axes.get_yaxis().set_ticks([])

plt.show()
data = {"X": X, "y": y}


def userInput():
    # Tao input 
    soLopInput = input("So lop: ")

    try: 
        # Chuyển soLopInput sang số nguyên  
        soLop = int(soLopInput) 
        # check số tự nhiên khác 0
        if soLop != float(soLopInput) or soLop <= 0:
            raise Exception()
    except:
        print("So khong hop le! (phai la so tu nhien khac 0)")
        # Lặp lại nếu lỗi
        return userInput()

    listNodeTrongLop = []

    i = 0
    
    while i < soLop:
        soNodeInput = input("Nhap so node o lop %d: " % i)

        try:
            soNode = int(soNodeInput) 
            # check số tự nhiên khác 0
            if soNode != float(soNodeInput) or soNode <= 0:
                raise Exception()
            listNodeTrongLop.append(soNode)
        except:
            print("So node khong hop le! (phai la so tu nhien khac 0)")
            # Lặp lại nếu lỗi
            continue

        i = i + 1

    # Trả về một object kiểu Dict. Lấy dữ liệu như từ điển, như object["soLop"]
    return {"soLop": soLop, "listNodeTrongLop": listNodeTrongLop}

networkConfig = userInput()

def mLPFsoftmax(V):
    e_V = np.exp(V - np.max(V, axis = 0, keepdims = True))
    Z = e_V / e_V.sum(axis = 0)
    return Z

## One-hot coding
def convert_labels(y, C = 3):
    Y = sparse.coo_matrix((np.ones_like(y),
        (y, np.arange(len(y)))), shape = (C, len(y))).toarray()
    return Y

# cost or loss function
l = 0.01
def mLPFcost(Y, Yhat, Wl, W):
    w = []
    for i in W:
        w.append(np.square(i).sum())
    wl = np.square(Wl).sum()
    R = sum(w) + wl
    return (-np.sum(Y*np.log(Yhat))/Y.shape[1])+(l/(2*Y.shape[1]))*R

def mLPTraining(X, y, soLop, listNodeTrongLop):
    d0 = 2
    # LayerCount là hidden layer count. Ta gọi dl là d của lớp cuối (output layer) 
    dl = C = 3
    # Tạo một danh sách w, sẽ kích thước là soLop
    W = []
    b = []

    i = 0
    while i <= soLop - 1:
        preNodeCount = d0
        if i != 0:
            preNodeCount = listNodeTrongLop[i - 1]
        W.append(0.01*np.random.randn(preNodeCount, listNodeTrongLop[i]))
        b.append(np.zeros((listNodeTrongLop[i], 1)))
        i = i + 1

    Wl = 0.01*np.random.randn(listNodeTrongLop[soLop - 1], dl)
    bl = np.zeros((dl, 1))

    Y = convert_labels(y, C)

    N = X.shape[1]
    eta = 1 # Tham chiếu learning rate
    for j in range(1000):
        ## Feedforward
        Z = []
        A = []
        
        i = 0
        while i < soLop:
            preA = X
            if i != 0:
                preA = A[i - 1]
            Z.append(np.dot(W[i].T, preA) + b[i])
            A.append(np.maximum(Z[i], 0))
            i = i + 1

        Zl = np.dot(Wl.T, A[soLop - 1]) + bl

        Yhat = mLPFsoftmax(Zl) #Softmax

        # In loss sau mỗi 1000 vòng lặp 
        if j %10 == 0:
            # average cross-entropy loss
            loss = mLPFcost(Y, Yhat, Wl, W)
            print("iter %d, loss: %f" %(j, loss))

        # backpropagation
        El = (Yhat - Y )/N
        dWl = np.dot(A[soLop - 1], El.T)
        dbl = np.sum(El, axis = 1, keepdims = True)

        EPre = El
        dW = []
        db = []

        i = soLop - 1
        while i >= 0:
            Wt = Wl
            if i < soLop - 1:
                Wt = W[i + 1]
            # EPre là E tính trước đó, EPre cập nhật sau mỗi vòng lặp 
            EPre = np.dot(Wt, EPre)
            EPre[Z[i] <= 0] == 0
            if i != 0:
                dW.append(np.dot(A[i - 1], EPre.T))
            else:
                dW.append(np.dot(X, EPre.T))
            db.append(np.sum(EPre, axis = 1, keepdims = True))
            i = i - 1

        # Đảo ngược mảng dW, do vòng lặp trước đi lùi về 0
        dW.reverse()
        db.reverse()

        # Cập nhật gradient Descent

        for i in range(0, soLop - 1):
            W[i] += -eta*dW[i]
            b[i] += -eta*db[i]
        
        Wl += -eta*dWl
        bl += -eta*dbl

    # Apre là A trước đó, tại i - 1 hoặc A1 (i == 0)
    Apre = X

    for i in range(0, soLop - 1):
        Z[i] = np.dot(W[i].T, Apre) + b[i]
        A[i] = np.maximum(Z[i], 0)
        Apre = A[i]

    Zl = np.dot(Wl.T, A[soLop - 1]) + bl

    predicted_class = np.argmax(Zl, axis=0)
    acc = (100*np.mean(predicted_class == y))
    print('training accuracy: %.2f %%' % acc)

    # Trực quan hóa kết quả phân loại 
    xm = np.arange(-1.5, 1.5, 0.025)
    xlen = len(xm)
    ym = np.arange(-1.5, 1.5, 0.025)
    ylen = len(ym)
    xx, yy = np.meshgrid(xm, ym)

    print(np.ones((1, xx.size)).shape)
    xx1 = xx.ravel().reshape(1, xx.size)
    yy1 = yy.ravel().reshape(1, yy.size)

    X0 = np.vstack((xx1, yy1))

    Z1 = np.dot(W[0].T, X0) + b[0] 
    A1 = np.maximum(Z1, 0)
    Z2 = np.dot(W[1].T, A1) + b[1]
    # predicted class 
    Zm = np.argmax(Z2, axis=0)

    Zm = Zm.reshape(xx.shape)
    CS = plt.contourf(xx, yy, Zm, 200, cmap='jet', alpha = .1)

    # X = X.T
    N = 100
    plt.plot(X[0, :N], X[1, :N], 'bs', markersize = 7);
    plt.plot(X[0, N:2*N], X[1, N:2*N], 'g^', markersize = 7);
    plt.plot(X[0, 2*N:], X[1, 2*N:], 'ro', markersize = 7);

    plt.xlim([-1.5, 1.5])
    plt.ylim([-1.5, 1.5])

    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.xticks(())
    plt.yticks(())
    plt.title('#hidden units = %d, accuracy = %.2f %%' %(listNodeTrongLop[0], acc))
    plt.show()

mLPTraining(data["X"], data["y"], networkConfig["soLop"], networkConfig["listNodeTrongLop"])