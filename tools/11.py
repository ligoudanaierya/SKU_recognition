import os
list_dir = os.listdir('cola')
for i,dir in enumerate(list_dir):
    print(i+24,dir)
    old_name = os.path.join('cola',dir)
    new_name = old_name.replace('fal_','')
    os.rename(old_name,new_name)