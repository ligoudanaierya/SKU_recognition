import tensorflow as tf
import present
from foward_p import forward
import KNN_P as KNN
import os
from sklearn.neighbors import KNeighborsClassifier

object_detect = present.present()
knn = KNeighborsClassifier(n_neighbors=1)
knn = KNN.create()
lenggui_dir = './tcp'
lenggui_list = os.listdir(lenggui_dir)
list2 = lenggui_list.copy()
for f in list2 :
	if os.path.isfile(os.path.join(lenggui_dir,f)):
		#print(os.path.join(lenggui_dir,f))
		lenggui_list.remove(f)
for lenggui_id in lenggui_list:
	lenggui_path = os.path.join(lenggui_dir, lenggui_id)#"./tcp/868575026467995/"
	file_name = os.listdir(lenggui_path)
	file_name.sort(reverse=True)
	#print(file_name)
	idx = 0
	flag = True
	while(flag and idx+1<len(file_name)):
		img1_name = os.path.join(lenggui_path, file_name[idx])
		img2_name = os.path.join(lenggui_path, file_name[idx+1])
		size1 = os.path.getsize(img1_name)
		size2 = os.path.getsize(img2_name)
		time1 = int(file_name[idx].split('.')[0].replace('-',''))
		time2 =int(file_name[idx+1].split('.')[0].replace('-',''))
		if size1>900000 and size2>900000 and time1-time2<500:
			print("start detect: ",time1-time2)
			print(img1_name,' : ',size1)
			print(img2_name," : ",size2)
			flag=False
			object_detect.demo(img1_name)
			object_detect.demo(img2_name)
			tf.reset_default_graph()
			forward(img1_name,img2_name)
			show_path,score,pre_num,all_num = KNN.pre(knn,img1_name, img2_name)
			print(show_path,score)
			
		else :
			idx = idx+1

	

