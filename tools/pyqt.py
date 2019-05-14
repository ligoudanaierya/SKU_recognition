import sys
import  os
sys.path.append("../")

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from present import present
import KNN_P
import time
from foward_p import forward
from sklearn.neighbors import KNeighborsClassifier
import tensorflow as tf
class picture(QWidget):
    def __init__(self):
        super(picture, self).__init__()
        self.knn = KNeighborsClassifier(n_neighbors=1)
        self.imgNameR=""
        self.imgNameL=""

        self.present = present()
        self.resize(1200, 1000)
        self.setFixedSize(self.width(), self.height())
        self.setWindowTitle("123")                        #文件title
        window_pale = QtGui.QPalette()
        window_pale.setBrush(self.backgroundRole(), QtGui.QBrush(QtGui.QPixmap("timg.jpg")))#整体背景填充
        self.setPalette(window_pale)
        self.setWindowIcon(QIcon('logo.jpg'))
        #self.show()


        text = QLabel(self)                                              #文字显示
        text.setText("蓝框表示可口可乐，红框表示非可口可乐")
        text.move(350, 40)
        text.setStyleSheet("QLabel{color:rgb(300,300,300,120);font-size:20px;font-weight:bold;font-family:宋体;}")


        pic = QPixmap('pic.jpg')                                        #原始图像区域填充图片背景路径

        self.pic1 = QLabel(self)                                         #原始图像（左边） 300*200大小  （50,100）为左上角坐标
        '''self.pic1.setText("冰柜检测图片显示")
        self.pic1.setFixedSize(320, 240)
        self.pic1.move(50, 100)
        self.pic1.setPixmap(pic)
        self.pic1.setStyleSheet("border: 2px solid red")
        self.pic1.setScaledContents(True)

        self.pic2 = QLabel(self)                                         #原始图像（右边）
        # lb2.setGeometry(0,250,300,200)
        self.pic2.setFixedSize(320, 240)
        self.pic2.move(400, 100)
        self.pic2.setPixmap(pic)
        self.pic2.setStyleSheet("border: 2px solid red")
        self.pic2.setScaledContents(True)'''


        '''btn_l = QPushButton(self)                                           #左边按钮点击打开图片
        btn_l.setText("打开图片")
        btn_l.move(300, 310)
        

        btn_l.clicked.connect(self.openimage_l)

        btn_r = QPushButton(self)                                           #右边按钮点击打开图片
        btn_r.setText("打开图片")
        btn_r.move(650, 310)'''
        

        #btn_r.clicked.connect(self.openimage_r)


        #btn_det = QPushButton(self)                                         #检测按钮调用另一个PY文件
        #btn_det.setText("执行检测")
        #btn_det.move(580, 530)
        #btn_det.clicked.connect(self.detection)

        '''btn_gj = QPushButton(self)                                         #chuangjian  anniu  调用另一个PY文件
        btn_gj.setText("Create KNN")
        btn_gj.move(300, 400)
        btn_gj.clicked.connect(self.create)'''


        self.newpic1 = QLabel(self)                                          #检测完成图像（左边）
        self.newpic1.setFixedSize(480, 360)
        self.newpic1.move(50, 100)
        self.newpic1.setScaledContents(True)

        self.newpic2 = QLabel(self)                                           #检测完成图像（右边）
        self.newpic2.setFixedSize(480, 360)
        self.newpic2.move(550, 100)
        self.newpic2.setScaledContents(True)

        self.score = QLabel(self)                                           #得分
        self.score.setText("‘千里眼’冷柜系统祝您生活愉快！                                               ")
        self.score.move(140, 600)
        self.score.setStyleSheet("QLabel{color:rgb(255,255,255,255);font-size:25px;font-weight:bold;font-family:宋体;}")
        self.knn = KNN_P.create()


    '''def openimage_l(self):
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.jpg;;*.png;;All Files(*)")

        
        jpg = QPixmap(imgName)
        self.imgNameL = imgName
        self.pic1.setPixmap(jpg)

    def openimage_r(self):
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.jpg;;*.png;;All Files(*)")
        jpg = QPixmap(imgName)
        self.imgNameR = imgName
        self.pic2.setPixmap(jpg)'''

    def detection(self,filename):
        filename_auto = "./tcp/868575026467995/"
        imgName = filename_auto+filename[-1]
        #jpg = QPixmap(imgName)
        self.imgNameL = imgName
        #self.pic1.setPixmap(jpg)
        imgName_r = filename_auto+filename[-2]
        #jpg_r = QPixmap(imgName_r)
        self.imgNameR = imgName_r
        #self.pic2.setPixmap(jpg_r)
        print(self.imgNameL,self.imgNameR)
        if self.imgNameL and self.imgNameR:
            self.present.demo(self.imgNameL)
            self.present.demo(self.imgNameR)
            tf.reset_default_graph()
            forward(self.imgNameL,self.imgNameR)
            show_path,score,pre_num,all_num = KNN_P.pre(self.knn,self.imgNameL,self.imgNameR)
                                                                   #**********************************************
            [self.NewimgNameL,self.NewimgNameR] = show_path[0],show_path[1]           #函数接口形式，参数有两个，为待检测图片的地址；返回值有三个，为保存的检测结果图片的地址和纯净度得分。
            newpicL = QPixmap(self.NewimgNameL)
                                                  #***********************************************
            newpicR = QPixmap(self.NewimgNameR)
            self.newpic1.setPixmap(newpicL)
            self.newpic1.setStyleSheet("border: 2px solid red")
            self.newpic1.setScaledContents(True)
            self.newpic2.setPixmap(newpicR)
            self.newpic2.setStyleSheet("border: 2px solid red")
            self.newpic2.setScaledContents(True)
            result_score = "可乐数量:"+str(pre_num)+"个 饮料总数:"+str(all_num)+"个 可乐占比:"+str(round(score,4))
            print(score)
            self.score.setText(result_score)

        else:
            QMessageBox.information(self, 'waring', '请选择待检测图像', QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)

    #def create(self):
        #print(sys.path)

        
import threading
def jiance(my):
    filename_auto = "./tcp/868575026467995/"
    
    num_images = 0
    
    while(True):
        filename=os.listdir(filename_auto)
        #print('lalalaalalalala')
        print("len(filename)-num_images: ",len(filename)-num_images)
        if (len(filename)-num_images)>1:
            filename.sort()
            img1_name = filename_auto+filename[-1]
            img2_name = filename_auto+filename[-2]
            size1 = os.path.getsize(img1_name)
            size2 = os.path.getsize(img2_name)
            time1 = int(filename[-1].split('.')[0].replace('-',''))
            time2 =int(filename[-2].split('.')[0].replace('-',''))
            print("size1 and size2 and time1-time2",size1,size2,time1-time2)
            if size1==921600 and size2==921600 and time1-time2<500:
                num_images = len(filename)
                my.detection(filename)
                
            
        
                
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    my = picture()
    my.show()
    t1 = threading.Thread(target = jiance,args=(my,))
    t1.setDaemon(True)
    t1.start()
    
    
        
    sys.exit(app.exec_())
    str()
