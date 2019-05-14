import os
import shutil
for root, dir, f in os.walk('middle_dir_2'):
	for p in f:
		path = os.path.join(root,p)
		if path.split(".")[-1]=="png":
			print(path)
			shutil.copyfile(path,'result/'+os.path.split(path)[-1])	
