
from sklearn.neighbors import KNeighborsClassifier
import glob
import numpy as np
import os
import re
import cv2
import matplotlib.pyplot as plt
import sys
#print(os.path.dirname(os.path.realpath(__file__)))
dir_path=(os.path.dirname(os.path.realpath(__file__)))
X = []
Y = []
print(sys.path)
for root, dir, f in os.walk(dir_path+'/all'):
    for p in f :
        path = os.path.join(root,p)
        if re.search('.npy', path) is not None:
            npy = np.load(path)
            #print(path)
            label = int(path.split('/')[-2])
            print(label)
            X.append(npy)
            Y.append(label)

X = np.squeeze(np.array(X))
Y = np.array(Y)
#print(Y)
print(X.shape,Y.shape)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X,Y)
print("Create KNN Finished!!!")
def pre():
    i_m = 0
    fal_num = 0
    all_num = 0
    #test_path = 'testdata/fal/00489.npy'
    #img_names =  glob.glob('test/test/*.npy')
    file_names  = os.listdir('middle_dir')
    res_dict = {}
    f_all = open('middle_dir/'+'result.txt', 'a')
    for f in file_names:
        i_m= i_m+1
        if f == 'result.txt':
             continue
        if re.search('_result.txt',f) is not None:
            continue
        txt_path = os.path.join('middle_dir',f,f.replace('.jpg','_result.txt'))
        map_path =txt_path.replace('_result.txt','_map.txt')

        if os.path.isfile(txt_path):
            continue
        f_o = open(txt_path, 'a')
        dir_name = os.path.join('middle_dir',f)
        file  = os.listdir(dir_name)
        #print('f:',f)
        all_num = 0
        fal_num = 0
        pre_num = 0
        npy_pre = {}
        for name in file:
            if re.search('.npy',name) is not None :
                all_num = all_num + 1
                npy_path = os.path.join(dir_name, name)
                test_npy = np.load(npy_path)
                pre = knn.predict(test_npy)
                prob = knn.predict_proba(test_npy)
                # print(np.max(np.array(prob)))

                nei = knn.kneighbors(test_npy, 10, True)
                if pre>32 or nei[0][0][0] > 28:
                    pre_end = 'fal'
                    fal_num += 1
                else:
                    pre_end = 'true'
                    pre_num += 1
                npy_pre[npy_path.split('/')[-1].replace('.npy', '')]=pre_end
        #print(type(npy_pre))        #
        #print(npy_pre)
        npy_pre2 = sorted(npy_pre.items(),key=lambda item:int(item[0].split('_')[0]))
        for key in npy_pre2:
            f_o.write(key[0] + '  '+key[1] + '\n')
        f_o.close()

        #npy_pre     0_03-19-09-42-04.jpg:true
        #map_txt     0_03-19-09-42-04.jpg:91.83241_444.12024_139.54962_591.8054
        f_map = open(map_path, 'r')
        jpg_path = os.path.join('testset', f)
        im = cv2.imread(jpg_path)
        im2 = im[:, :, (2, 1, 0)]
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(im2, aspect='equal')
        for line in f_map.readlines():
            line = line.rstrip('\n')
            cut_name = line.split(':')[0]
            bbox = line.split(':')[1].split('_')
            bbox = [float(i) for i in bbox]
            if npy_pre[cut_name] == 'true':
                ax.add_patch(
                    plt.Rectangle((bbox[0], bbox[1]),
                                  bbox[2] - bbox[0],
                                  bbox[3] - bbox[1], fill=False,
                                  edgecolor='blue', linewidth=3.5)
                )
            else:
                ax.add_patch(
                    plt.Rectangle((bbox[0], bbox[1]),
                                  bbox[2] - bbox[0],
                                  bbox[3] - bbox[1], fill=False,
                                  edgecolor='red', linewidth=3.5)
                )
        f_map.close()
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join('middle_dir',f,f.replace('.jpg','.png')))

        #print('aaaaa', npy_pre)
        #print(type(npy_pre))

        precision = float(pre_num) / all_num
        res_dict[f]=':  '+'可口可乐的数量： '+str(pre_num)+', 总的饮料数量： '+str(all_num)+', 可口可乐占比为： '+str(precision)+'\n'
        
        print(i_m)
        
        #f_all.write(f+':  '+'可口可乐的数量： '+str(pre_num)+', 总的饮料数量： '+str(all_num)+', 可口可乐占比为： '+str(precision)+'\n')

        
    #res_dict = sorted(res_dict.items(),key=lambda item:int(item[0].split('_')[0]))
    print(res_dict,'\n')
    for key in res_dict.keys():
        #print(key[0],key[])
        f_all.write(key + res_dict[key] + '\n')
    f_all.close()
'''import cv2
import matplotlib.pyplot as plt
for root, dir, f in os.walk('middle_dir/video2.jpg'):
    for p in f:
        path = os.path.join(root, p)
        #txt_path = 'middle_dir/' + path.split('/')[1] + '/' + 'result.txt'
        #f = open(txt_path, 'a')
        if re.search('.npy',path) is not None :
            #all_num += 1
            test_npy = np.load(path)

#for test_path in img_names:
    #all_num += 1
#test_path = 'cola/1/0.npy'
    #test_npy = np.load(test_path)
    #test_npy = np.squeeze(test_npy)

            pre = knn.predict(test_npy)
            #prob = knn.predict_proba(test_npy)
            #print(prob)
            #print(np.max(np.array(prob)))
            print('pre:',pre)
            nei = knn.kneighbors(test_npy,10,True)
            if pre>32 or nei[0][0][0]>30:
                pre_end = 'fal'
            else :
                pre_end = 'true'

            #f.write(l.split('/')[-1].replace('.npy', '  ')+str(pre)+'\n')
            #f.close
            #print(pre==Y)
            #if pre != '2':
                #fal_num += 1

            print(path)
            print(pre)
            print(pre_end)
            #print(prob)
            print(nei[0][0])
            if pre_end=='true':
                img = cv2.imread(path.replace('.npy', ''))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                plt.imshow(img)
                plt.show()'''
