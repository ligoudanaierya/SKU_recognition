import cv2

def det(filename1,fliename2):
    image = cv2.imread(filename1)
    # size = image.shape
    # print(self.imgNameR)
    # cv2.imwrite("MyPic.jpg",image)
    file_path1 = file_name1
    file_path2 = "pic.jpg"
    score = 100
    return file_path1,file_path2,score
