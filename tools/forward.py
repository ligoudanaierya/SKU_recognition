import numpy as np
import cv2
import tensorflow as tf
import re
from resnet import *
import  matplotlib.pyplot as plt
import glob
def forward():
	input = tf.placeholder(dtype=tf.float32,shape=(1,224,224,3))
	is_training = tf.placeholder('bool', [], name='is_training')
	prob,avg= inference(input,is_training=is_training,num_classes=62,bottleneck=True,num_blocks=[3, 4, 23, 3])
	na = tf.nn.softmax(prob)
	#sess.run(tf.global_variables_initializer())
	train  = tf.trainable_variables()
	g_var = tf.global_variables()
	for g in g_var:
		if re.search('moving_mean:0|moving_variance:0*',g.name) is not None:
			#print(g.name)
			train += [g]
	saver = tf.train.Saver(train)
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	saver.restore(sess,'./101_checkpoint/model.ckpt-1501')

	label_map={}
	'''dir = glob.glob('cola'+ '/*')
	for i, label_name in enumerate(dir):
		label_map[label_name.split('/')[1]] = {'index': i, 'desc': label_name.split('/')[1]}'''

	#print(label_map)
	for i in os.listdir('all/48/'):
		path = os.path.join('all/48',i)
		if path.split('.')[-1]=='jpg':
				print(path)
				img = cv2.imread(path)
				img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
				img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
				# plt.imshow(img)
				# plt.show()
				img = np.reshape(img, (1, 224, 224, 3))

				img = img / 255.
				feature = sess.run(avg, feed_dict={input: img,is_training:False})
				#label = int(path.split('/')[1])
				np.save(path,feature)
			
	'''for root, dir, f in os.walk('middle_dir'):
		for p in f :
			

			path = os.path.join(root,p)
			if path.split('.')[-1]=='jpg':
				print(path)
				img = cv2.imread(path)
				img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
				img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
				# plt.imshow(img)
				# plt.show()
				img = np.reshape(img, (1, 224, 224, 3))

				img = img / 255.
				feature = sess.run(avg, feed_dict={input: img,is_training:False})
				#label = int(path.split('/')[1])
				np.save(path,feature)'''


'''p_num =0
all_num = 0
img_names = glob.glob('testdata/*.jpg')
for img_name in img_names:

	all_num += 1
	if re.search('_', img_name) is  None:
		label_index = 4
	else :
		label_index = label_map[img_name.split('_')[-1].split('.')[0]]['index']
	#print(img_name, label_index)
	img = cv2.imread(img_name)
	img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
	img = cv2.resize(img,(224,224),interpolation=cv2.INTER_CUBIC)
	#plt.imshow(img)
	#plt.show()
	img = np.reshape(img,(1,224,224,3))


	img = img/255.
	p =sess.run(na, {input:img,is_training:False})
	p = np.argmax(p,1)
	if p == label_index:
		p_num+=1
	else: print(img_name, label_index, p)
print('p_num:',p_num,'all_num:',all_num,'p:',p_num/all_num)'''

'''saver = tf.train.import_meta_graph('./101_checkpoint/model.ckpt-501.meta')
graph = tf.get_default_graph()


tensor_name_list = [tensor.name for tensor in graph.as_graph_def().node]
for i in range(len(tensor_name_list)):
    if re.search('fc',tensor_name_list[i]) is not None:
        print(tensor_name_list[i])'''
'''prob_tensor = graph.get_tensor_by_name("prob:0")
images = graph.get_tensor_by_name("images:0")
for op in graph.get_operations():
    print (op.name)

#init = tf.initialize_all_variables()
#sess.run(init)
print ("graph restored")

batch = img.reshape((1, 224, 224, 3))

feed_dict = {images: batch}

prob = sess.run(prob_tensor, feed_dict=feed_dict)

'''
forward()
