import numpy as np
import matplotlib.pyplot as plt
 # Loss function computation
def find_loss(x,y,m1,m2):
    predict=m1*x + m2
    return np.mean((predict-y)**2)
#Gradient descent implementation
def gradient_descent(x, y, learning_rate, epochs):
    n=len(x)
    m1=np.random.randn()
    m2=np.random.randn()
    loss_array=[]

    for epoch in range(epochs):

        y_pred=m1*x + m2
        error=y-y_pred
        
        dm1= (-2/n) * np.sum(x*error)
        dm2= (-2/n) * np.sum(error)

        m1-=learning_rate*dm1
        m2-=learning_rate*dm2
        
        loss=find_loss(x,y,m1,m2)
        loss_array.append(loss)
        if epoch%100==0:
            print(f"Epoch {epoch}: loss={loss},m1={m1},m2={m2}")
    return m1,m2,loss_array   
#Best line fit plot
def plot_best_fit_line(x,y,m1,m2):
    plt.figure(figsize=(8,4))
    plt.scatter(x,y,label="Data with noise",color="blue")
    plt.plot(x,m1*x+m2,label="Best fit line",color="red")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()
#loss function plot
def plot_loss(loss_array):
    plt.plot(loss_array)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Gradient Descent Loss Function')
    plt.show()
#generating synthetic data
np.random.seed(42)
m1_true=3
m2_true=2
x=np.random.randn(100,1)
noise=np.random.normal(0,1,(100,1))
y=m1_true*x + m2_true + noise
epochs=5000
learning_rate=0.01
#function calls
m1,m2,loss_array=gradient_descent(x,y,learning_rate,epochs)
plot_best_fit_line(x,y,m1,m2)
plot_loss(loss_array)
