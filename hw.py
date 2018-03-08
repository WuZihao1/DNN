import tensorflow as tf
import numpy as np 
from PIL import Image
import os
import random

filepath='./train_reshaped/train_reshaped'
list = os.listdir(filepath)
current_index = 0
image_set = np.zeros((len(list),100,100,3),dtype=np.float32)
label_set = np.zeros((len(list),25))

content = open("train_label.txt","r")
line = content.readlines()
random.shuffle(line)

for i in range(len(line)):
	name,label = line[i].split(',')
	img = np.asarray(Image.open(filepath+"/"+name))
	image_set[i] = img
	label_set[i] = np.zeros(25)
	label_set[i][int(label)] = 1

np.savez("train_set.npy",images=image_set,labels=label_set)
content.close()


train_set = np.load("train_set.npz")
test_set = np.load("validation.npy").item()


imgs = train_set["images"]
labels = train_set["labels"]
imgs_size = imgs.shape[0]

def get_train_batch(batch_size):
  global current_index
  
  if current_index*batch_size+batch_size>imgs_size:
    batch_imgs = np.zeros((batch_size,100,100,3))
    batch_imgs[0:imgs_size-batch_size*current_index] = imgs[current_index*batch_size:imgs_size]
    y_label = np.zeros((batch_size,25))
    y_label[0:imgs_size-batch_size*current_index] = labels[current_index*batch_size:imgs_size]
    batch_imgs[imgs_size-batch_size*current_index:]=imgs[0:batch_size-imgs_size+batch_size*current_index]
    y_label[imgs_size-batch_size*current_index:] = labels[0:batch_size-imgs_size+batch_size*current_index]
    current_index=0
    return batch_imgs, y_label
    
  batch_imgs = np.array(imgs[current_index*batch_size:current_index*batch_size+batch_size])
  y_label = np.array(labels[current_index*batch_size:current_index*batch_size+batch_size])
  current_index += 1
  return batch_imgs, y_label
    
    




def compute_accuracy(v_xs,v_ys):
	global prediction
	y_pre = sess.run(prediction,feed_dict={xs:v_xs,keep_prob:1})
	correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
	result = sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys,keep_prob:1})
	return result


def weight_variable(shape):
	initial = tf.truncated_normal(shape,stddev=0.1)
	return tf.Variable(initial)


def bias_variable(shape):
	initial = tf.constant(0.1,shape=shape)
	return tf.Variable(initial)

def conv2d(x,W):
	return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')




xs = tf.placeholder(tf.float32,[None,100,100,3])
ys = tf.placeholder(tf.float32,[None,25])
keep_prob = tf.placeholder(tf.float32)

W_conv1 = weight_variable([5,5,3,32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(xs,W_conv1)+b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_conv3 = weight_variable([5,5,64,128])
b_conv3 = bias_variable([128])
h_conv3 = tf.nn.relu(conv2d(h_pool2,W_conv3)+b_conv3)
h_pool3 = max_pool_2x2(h_conv3)

W_fc1 = weight_variable([13*13*128,1024])
b_fc1 = bias_variable([1024])
h_pool3_flat = tf.reshape(h_pool3,[-1,13*13*128])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat,W_fc1)+b_fc1)
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)

W_fc2 = weight_variable([1024,25])
b_fc2 = bias_variable([25])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)
test_img = np.array(test_set["reshaped"]).reshape(len(test_set["reshaped"]),100,100,3).astype("uint8")
test_label = np.array(test_set["label"])
test_labels = np.zeros((250,25))
for i in range(250):
  test_labels[i][test_label[i]]=1

cross_entropy=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(2000):
  batch_xs, batch_ys = get_train_batch(50)
  sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys,keep_prob:0.5})
  if i % 50 == 0:
    print(compute_accuracy(test_img,test_labels))
