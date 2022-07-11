import numpy as np
import matplotlib.pyplot as plt

def generate_linear(n=100):
    pts = np.random.uniform(0 ,1 , (n, 2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        distance = (pt[0]-pt[1])/1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    
    return np.array(inputs), np.array(labels).reshape(n, 1)

def generate_XOR_easy():
    inputs = []
    labels = []

    for i in range(11):
        inputs.append([0.1*i, 0.1*i])
        labels.append(0)

        if 0.1*i == 0.5:
            continue
        
        inputs.append([0.1*i, 1-0.1*i])
        labels.append(1)
    
    return np.array(inputs), np.array(labels).reshape(21, 1)

def show_result(x, y, pred_y):
    print("size of x", x.shape)
    print("size of y", y.shape)
    print("size of pred", np.array(pred_y).shape)
    plt.subplot(1,2,1)
    plt.title('Ground truth', fontsize=18)
    for i in range(x.shape[0]):
        if(y[i]==0):
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')

    plt.subplot(1,2,2)
    plt.title('Predict result', fontsize=18)
    for i in range(x.shape[0]):
        if(pred_y[i]==0):
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')

    plt.savefig("result_XOR.png")

class Adam:
    def __init__(self, eta = 0.05, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.m_dw, self.v_dw = 0, 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.eta = eta
        self.t = 1

    def init_para(self):
        self.m_dw, self.v_dw = 0, 0

    def update(self, w, dw):
        self.init_para()
        self.m_dw = self.beta1*self.m_dw + (1-self.beta1)*dw
        
        ## rms beta 2
        self.v_dw = self.beta2*self.v_dw + (1-self.beta2)*(dw**2)

        ## bias correction
        m_dw_corr = self.m_dw/(1-self.beta1**self.t)
        v_dw_corr = self.v_dw/(1-self.beta2**self.t)

        ## update weights and biases
        w = w - self.eta*(m_dw_corr/(np.sqrt(v_dw_corr)+self.epsilon))
        return w

class GD:
    def __init__(self, eta = 0.05):
        self.eta = eta

    def update(self, w, dw):
        w = w - self.eta * dw
        return w

class MLP:
    def __init__(self, hidden_size1=10, hidden_size2=10):
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2

        self.w1 = np.random.rand(2, self.hidden_size1)
        self.w2 = np.random.rand(self.hidden_size1, self.hidden_size2)
        self.w3 = np.random.rand(self.hidden_size2, 1)
        self.optimizer = GD()

        self.inputs = np.zeros((2,1))
        self.output1 = np.zeros((self.hidden_size1, 1))
        self.output2 = np.zeros((self.hidden_size2, 1))
        self.y = np.array([0])

        self.grad3 = np.array([0])
        self.grad2 = np.zeros((self.hidden_size2, 1))
        self.grad1 = np.zeros((self.hidden_size1, 1))

        self.dw3 = np.zeros((self.hidden_size2, 1))
        self.dw2 = np.zeros((self.hidden_size1, self.hidden_size2))
        self.dw1 = np.zeros((2, self.hidden_size1))

    def sigmoid(self, x):
        return np.divide(1.0, (1.0+np.exp(-x)))

    def ReLU(self, x):
        return (abs(x) + x) / 2

    def de_ReLU(self, x):
        return 1 * (x > 0)
    
    def de_BCE(self, y_true, y):
        return (y - y_true)/(y*(1-y) + 1e-9)
 
    def de_sigmoid(self, x):
        return np.multiply(x, 1.0-x)

    def forward(self, x):
        self.inputs = x.reshape(-1,1)
        # print("size of input: ", self.inputs.shape)
        h1 = np.matmul(np.transpose(self.w1), self.inputs)
        self.output1 = self.sigmoid(h1).reshape(-1,1)
        # print("size of output1: ", self.output1.shape)
        h2 = np.matmul(np.transpose(self.w2), self.output1)
        self.output2 = self.sigmoid(h2).reshape(-1,1)
        # print("size of output2: ", self.output2.shape)
        h3 = np.matmul(np.transpose(self.w3), self.output2)
        self.y = self.sigmoid(h3).reshape(-1,1)

        return self.y

    def backward(self, y_true):
        self.grad3 = self.de_BCE(y_true, self.y) * self.de_sigmoid(self.y)
        self.dw3 = self.output2 * self.grad3
        # print("grad3 ", self.grad3)
        # print("dw3 ", self.dw3)

        self.grad2 = self.w3 * self.de_sigmoid(self.output2) * self.grad3
        # print("shape grad2 ", self.w3.shape)
        self.dw2 = np.matmul(self.output1, np.transpose(self.grad2))
        # print("shape dw2 ", self.dw2.shape)

        self.grad1 = np.matmul(self.w2 * self.de_sigmoid(self.output1), self.grad2)
        # print("shape grad1 ", self.grad1.shape)
        self.dw1 = np.matmul(self.inputs, np.transpose(self.grad1))

    # def forward(self, x):
    #     self.inputs = x.reshape(-1,1)
    #     # print("size of input: ", self.inputs.shape)
    #     self.output1 = np.matmul(np.transpose(self.w1), self.inputs).reshape(-1,1)
    #     # print("size of output1: ", self.output1.shape)
    #     self.output2 = np.matmul(np.transpose(self.w2), self.output1).reshape(-1,1)
    #     # print("size of output2: ", self.output2.shape)
    #     self.y =  np.matmul(np.transpose(self.w3), self.output2).reshape(-1,1)

    #     return self.y

    # def backward(self, y_true):
    #     self.grad3 = self.de_BCE(y_true, self.y)
    #     self.dw3 = self.output2 * self.grad3
    #     # print("grad3 ", self.grad3)
    #     # print("dw3 ", self.dw3)

    #     self.grad2 = self.w3 * self.grad3
    #     # print("shape grad2 ", self.w3.shape)
    #     self.dw2 = np.matmul(self.output1, np.transpose(self.grad2))
    #     # print("shape dw2 ", self.dw2.shape)

    #     self.grad1 = np.matmul(self.w2, self.grad2)
    #     # print("shape grad1 ", self.grad1.shape)
    #     self.dw1 = np.matmul(self.inputs, np.transpose(self.grad1))

    def update(self):
        self.w1 = self.optimizer.update(self.w1, self.dw1)
        self.w2 = self.optimizer.update(self.w2, self.dw2)
        self.w3 = self.optimizer.update(self.w3, self.dw3)




def BCEloss(output, target):
    n = len(output)
    # total_loss = 0
    total_loss = target * np.log(output + 1e-9) + (1-target)*np.log(1-output + 1e-9)
    
    return -1*total_loss


if __name__ == '__main__':

    np.random.seed(320)
    # x_train, y_train = generate_linear(n=100) # lr:0.05 epoch:3000
    x_train, y_train = generate_XOR_easy() # lr: 0.05 / epoch:5000

    model = MLP(5,5)
    epochs = 5000
    lr = 5
    loss_log=[]
    # model training
    for epoch in range(epochs):
        for x, y in zip(x_train, y_train):
            pred = model.forward(x)
            # print(pred)
            loss = BCEloss(y, pred)
            model.backward(y)
            model.update()
            # model.init_para()

        if((epoch+1)%10==0):
            loss_log.append(loss)

        if((epoch+1)%100==0):
            print(f"epoch: {epoch+1} -> training loss: {loss}")
    
    iteration = [10*x for x in range(len(loss_log))]
    plt.plot(iteration, np.array(loss_log).reshape(-1))
    plt.title("learning curve")
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.savefig('learning_curve.png')

    # testing 
    result = []
    for x in x_train:
        pred = model.forward(x)
        print(pred)
        if(pred>=0.5):
            result.append(1)
        else:
            result.append(0)

    acc = (result==y_train.reshape(-1)).sum()/len(y_train)
    print(f'correct classified sample / total samples = {(result==y_train.reshape(-1)).sum()} / {len(y_train)}')
    print(f'accuracy: {acc*100}%')

    # show result 
    show_result(x_train, y_train, result)




