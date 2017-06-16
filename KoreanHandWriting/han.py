
import h5py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
#
# sess = tf.InteractiveSession()
# sess.run(tf.global_variables_initializer())
# saver = tf.train.Saver()
# saver.restore(sess, './weight')


#print('reload has been done')



def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

with h5py.File('kalph_train.hf','r') as hf:
    images = np.array(hf['images'])
    labels = np.array(hf['labels'])
    num_imgs, rows, cols = images.shape


tmp=[]

for i in range(num_imgs):
    if labels[i]==0 :
        tmp.append([1,0,0,0,0,0,0,0,0,0,0,0,0,0])
    elif labels[i]==1 :
        tmp.append([0,1,0,0,0,0,0,0,0,0,0,0,0,0])
    elif labels[i]==2 :
        tmp.append([0,0,1,0,0,0,0,0,0,0,0,0,0,0])
    elif labels[i]==3 :
        tmp.append([0,0,0,1,0,0,0,0,0,0,0,0,0,0])
    elif labels[i]==4 :
        tmp.append([0,0,0,0,1,0,0,0,0,0,0,0,0,0])
    elif labels[i]==5 :
        tmp.append([0,0,0,0,0,1,0,0,0,0,0,0,0,0])
    elif labels[i]==6 :
        tmp.append([0,0,0,0,0,0,1,0,0,0,0,0,0,0])
    elif labels[i]==7 :
        tmp.append([0,0,0,0,0,0,0,1,0,0,0,0,0,0])
    elif labels[i]==8 :
        tmp.append([0,0,0,0,0,0,0,0,1,0,0,0,0,0])
    elif labels[i]==9 :
        tmp.append([0,0,0,0,0,0,0,0,0,1,0,0,0,0])
    elif labels[i]==10 :
        tmp.append([0,0,0,0,0,0,0,0,0,0,1,0,0,0])
    elif labels[i]==11 :
        tmp.append([0,0,0,0,0,0,0,0,0,0,0,1,0,0])
    elif labels[i]==12 :
        tmp.append([0,0,0,0,0,0,0,0,0,0,0,0,1,0])
    else :
        tmp.append([0,0,0,0,0,0,0,0,0,0,0,0,0,1])



label=np.array(tmp)

#images.dtype=np.float32
#labels.dtype=np.float32
label=label.reshape([19600,14])
images=np.reshape(images,[num_imgs,cols*rows])
# img=[]
#
# for i in range(0,num_imgs):
#     img.append(images[i].reshape(1,52*52))


x_input = tf.placeholder(tf.float32, shape=[None, rows*cols])
x_img=tf.reshape(x_input,[-1,52,52,1])
y_input = tf.placeholder(tf.float32, shape=[None, 14])
#Layer1
W1=tf.Variable(tf.random_normal([5,5,1,16],stddev=0.01)) #필터 16개
L1=tf.nn.conv2d(x_img,W1,strides=[1,1,1,1],padding='VALID')     #48
L1=tf.nn.relu(L1)
L1=tf.nn.max_pool(L1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')  #24
#Layer2
W1_1=tf.Variable(tf.random_normal([5,5,16,32],stddev=0.01))
L1_1=tf.nn.conv2d(L1,W1_1,strides=[1,1,1,1],padding='SAME')            #24
L1_1=tf.nn.relu(L1_1)
L1_1=tf.nn.max_pool(L1_1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')  #12
#Layer3
W2=tf.Variable(tf.random_normal([3,3,32,64],stddev=0.01))
L2=tf.nn.conv2d(L1_1,W2,strides=[1,1,1,1],padding='SAME')
L2=tf.nn.relu(L2)
L2=tf.nn.max_pool(L2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME') #6
L2=tf.reshape(L2,[-1,6*6*64])

#full c


W3_1=weight_variable(shape=[6*6*64,128])
b_1=tf.Variable(tf.random_normal([128]))
hypothesis1=tf.matmul(L2,W3_1)+b_1

keep_prob = tf.placeholder(tf.float32)
W3=weight_variable(shape=[128,14])
b=tf.Variable(tf.random_normal([14]))
hypothesis=tf.matmul(hypothesis1,W3)+b
h_fc1_drop = tf.nn.dropout(hypothesis, keep_prob)

cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis,labels=y_input))
optimizer=tf.train.AdamOptimizer(1e-4).minimize(cost)
correct_prediction=tf.equal(tf.argmax(hypothesis,1),tf.argmax(y_input,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))



sess=tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
batch_size=100
for i in range(0,19000):
    indices=np.random.choice(num_imgs,batch_size)
    x_batch,y_batch=images[indices],label[indices]
    train_accuracy=accuracy.eval(feed_dict ={x_input:x_batch,y_input:y_batch,keep_prob:1.0})

    optimizer.run(feed_dict={x_input:x_batch,y_input:y_batch,keep_prob:0.5})
    #if i%100==0:
    print('step=',i,' training accuracy=',train_accuracy)



with h5py.File('kalph_test.hf','r') as hf:
    images = np.array(hf['images'])
    labels = np.array(hf['labels'])
    num_imgs, rows, cols = images.shape


tmp=[]

for i in range(num_imgs):
    if labels[i]==0 :
        tmp.append([1,0,0,0,0,0,0,0,0,0,0,0,0,0])
    elif labels[i]==1 :
        tmp.append([0,1,0,0,0,0,0,0,0,0,0,0,0,0])
    elif labels[i]==2 :
        tmp.append([0,0,1,0,0,0,0,0,0,0,0,0,0,0])
    elif labels[i]==3 :
        tmp.append([0,0,0,1,0,0,0,0,0,0,0,0,0,0])
    elif labels[i]==4 :
        tmp.append([0,0,0,0,1,0,0,0,0,0,0,0,0,0])
    elif labels[i]==5 :
        tmp.append([0,0,0,0,0,1,0,0,0,0,0,0,0,0])
    elif labels[i]==6 :
        tmp.append([0,0,0,0,0,0,1,0,0,0,0,0,0,0])
    elif labels[i]==7 :
        tmp.append([0,0,0,0,0,0,0,1,0,0,0,0,0,0])
    elif labels[i]==8 :
        tmp.append([0,0,0,0,0,0,0,0,1,0,0,0,0,0])
    elif labels[i]==9 :
        tmp.append([0,0,0,0,0,0,0,0,0,1,0,0,0,0])
    elif labels[i]==10 :
        tmp.append([0,0,0,0,0,0,0,0,0,0,1,0,0,0])
    elif labels[i]==11 :
        tmp.append([0,0,0,0,0,0,0,0,0,0,0,1,0,0])
    elif labels[i]==12 :
        tmp.append([0,0,0,0,0,0,0,0,0,0,0,0,1,0])
    else :
        tmp.append([0,0,0,0,0,0,0,0,0,0,0,0,0,1])



label=np.array(tmp)

#images.dtype=np.float32
#labels.dtype=np.float32
label=label.reshape([3920,14])
images=np.reshape(images,[num_imgs,cols*rows])

test_accuracy=accuracy.eval(feed_dict={x_input:images,y_input:label,keep_prob:1.0})
print('test accuracy=',test_accuracy)


#------------------------------------------------------------------------------------------------