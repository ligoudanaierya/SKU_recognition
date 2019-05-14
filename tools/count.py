import os
import numpy as np

f1 = open('1.txt','r')
f2 = open('2.txt','r')
try:
	text_lines1 =f1.readlines()
	text_lines2 = f2.readlines()
	nums = 0
	for i in range(len(text_lines1)):
		line1 = text_lines1[i]
		line2 = text_lines2[i]
		line1 = line1.replace('\n','')
		line2 = line2.replace('\n','')
		count1 = float(line1)
		count2 = float(line2)
		if (count1>0.4 and count2<0.4) or (count1<0.4 and count2>0.4):
			nums = nums+1
			print(i,count1,count2)
	print(nums)
finally:
	f1.close()
	f2.close() 
			
