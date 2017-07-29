from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Add, Input, Lambda
from keras import optimizers
from keras import losses
from keras import backend as K
from keras.engine.topology import Layer
import tensorflow as tf
import numpy as np

class NoiseGenerate:
    def __init__(self):
        self.rnge = 1

    def sample(self, N , batch_size):
    	samples = np.array([ np.linspace(-self.rnge, self.rnge, N) + np.random.random(N) * 0.01 for i in range(batch_size) ])
    	labels = np.zeros((batch_size,1))
        return samples,labels

class TruthGenerate:
    def __init__(self):
        self.mu = 4
        self.sigma = 0.5
        
    def sample(self, N, batch_size):
    	samples = []
    	for i in range( batch_size ):
    		slic = np.random.normal(self.mu, self.sigma, N)
    		slic.sort()
    		samples.append(slic)
    	labels = np.ones((batch_size,1))
        return np.array(samples) , labels
        
tru = Input(shape=(100,))
nse = Input(shape=(100,))

x1 = Dense(500, input_shape=(100,))(nse)
x1 = Activation('softplus')(x1)
h1 = Dense(100)(x1)
generator = Model(inputs = nse, outputs = h1)

nse_in = Input(shape=(100,))

dense1 = Dense(1000, input_shape=(100,))
activation1 = Activation('tanh')
dense2 = Dense(1000)
activation2 = Activation('tanh')
dense3 = Dense(1000)
activation3 = Activation('tanh')
dense4 = Dense(1)
activation4 = Activation('sigmoid')

dd1 = dense1(nse_in)
dd1 = activation1(dd1)
dd1 = dense2(dd1)
dd1 = activation2(dd1)
dd1 = dense3(dd1)
dd1 = activation3(dd1)
dd1 = dense4(dd1)
r1 = activation4(dd1)

dd2 = dense1(tru)
dd2 = activation1(dd2)
dd2 = dense2(dd2)
dd2 = activation2(dd2)
dd2 = dense3(dd2)
dd2 = activation3(dd2)
dd2 = dense4(dd2)
r2 = activation4(dd2)


dd3 = dense1(h1)
dd3 = activation1(dd3)
dd3 = dense2(dd3)
dd3 = activation2(dd3)
dd3 = dense3(dd3)
dd3 = activation3(dd3)
dd3 = dense4(dd3)
r3 = activation4(dd3)

discriminator = Model(inputs=[nse_in,tru], outputs = [r1,r2])

gan = Model(inputs=[nse], outputs = [r3])


sgd = optimizers.SGD(lr=0.005, decay=0.95 )



def loss1(y_true, y_pred):
    #print y_pred.shape
    _loss = -tf.reduce_mean(tf.log(y_pred))#noise -> 1
    return _loss
def loss2(y_true, y_pred):
    #print y_pred.shape
    #_loss = tf.reduce_sum(tf.log(1-y_pred[0]))-tf.reduce_sum(tf.log(y_pred[1]))
    #print tf.reduce_sum(tf.log(1-y_pred[0]))
    return 0*y_pred
generator.compile(loss = loss1,optimizer=sgd)
discriminator.compile(loss='mean_squared_error',optimizer=sgd)
gan.compile(loss=loss1,optimizer=sgd)

NG = NoiseGenerate()
TG = TruthGenerate()



for i in range(500):
    dataN, labelN = NG.sample(100,1)
    dataT, labelT = TG.sample(100,1)
    dataH = generator.predict(dataN)
    discriminator.fit([dataH,dataT],[labelN,labelT],epochs=1,batch_size=1)

discriminator.add_loss(-tf.reduce_mean(tf.log(1-r1))-tf.reduce_mean(tf.log(r2)))#noise -> 0; data -> 1
discriminator.compile(loss=loss2, optimizer=sgd)
gan.compile(loss=loss1,optimizer=sgd)

for j in range(50):
    for ind in range(4):
        discriminator.layers[ind*2+2].trainable = True
    discriminator.compile(loss=loss2,optimizer=sgd)
    gan.compile(loss=loss1,optimizer=sgd)
    for i in range(50):
        dataN, _l = NG.sample(100,1)
        dataT, _l = TG.sample(100,1)
        dataH = generator.predict(dataN)
        discriminator.fit([dataH,dataT],[dataH,dataT],epochs=15,batch_size=1)
    for ind in range(4):
        discriminator.layers[ind*2+2].trainable = False
    discriminator.compile(loss=loss2,optimizer=sgd)
    gan.compile(loss=loss1,optimizer=sgd)
    for i in range(25):
        dataN, _l = NG.sample(100,1)
        dataT, _l = TG.sample(100,1)
        #dataH = generator.predict(dataN)
        gan.fit([dataN],[dataT],epochs=15,batch_size=100)

import matplotlib.pyplot as plt
import scipy.interpolate
dataN, _l = NG.sample(100,1)
result = generator.predict(dataN)
p, x = np.histogram(result, bins=100) # bin it into n = N/10 bins
x = x[:-1] + (x[1] - x[0])/2   # convert bin edges to centers
f = scipy.interpolate.UnivariateSpline(x, p, s = 1000)
plt.plot(x, f(x))
plt.show()