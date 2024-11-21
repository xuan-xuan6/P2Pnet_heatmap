import os
import json
dataset_root_path = r'D:\dataset\计数\crop_P2P'
#================生成list文件================#
# #读取train文件夹下的所有.bmp后缀的文件
train_list = os.listdir(os.path.join(dataset_root_path, 'train'))
test_list = os.listdir(os.path.join(dataset_root_path, 'test'))
# # # #去除train_list中不是.bmp后缀的文件
train_list = [item for item in train_list if item.endswith('.bmp')]
test_list = [item for item in test_list if item.endswith('.bmp')]
with open(os.path.join(dataset_root_path, 'train.txt'), 'w') as f:
    #遍历train_list,写入每个文件相对于train.txt的路径+空格+train文件夹下同名txt
    for item in train_list:
        f.write(os.path.join('train', item) + ' '+os.path.join('train', item.replace(".bmp",".txt"))+'\n')
with open(os.path.join(dataset_root_path, 'test.txt'), 'w') as f:
    #遍历train_list,写入每个文件相对于train.txt的路径+空格+train文件夹下同名txt
    for item in test_list:
        f.write(os.path.join('test', item) + ' '+os.path.join('test', item.replace(".bmp",".txt"))+'\n')
#==============生成txt文件=================#
# train_json_list = [item for item in train_list if item.endswith('.json')]
# test_json_list = [item for item in test_list if item.endswith('.json')]
# #遍历train_json_list,创建同名的txt文件，并将json文件中每个shapes内容写入
# for item in train_json_list:
#     with open(os.path.join(dataset_root_path, 'train', item.replace(".json",".txt")), 'w') as f:
#         with open(os.path.join(dataset_root_path, 'train', item), 'r') as json_file:
#             json_data = json.load(json_file)
#             for shape in json_data['shapes']:
#                 #遍历shapes中的项，获取到points内容中两个点的中心点坐标
#                 x = (shape['points'][0][0] + shape['points'][1][0])/2
#                 y = (shape['points'][0][1] + shape['points'][1][1])/2
#                 #将中心点坐标写入txt文件
#                 f.write(str(x) + ' '+str(y)+'\n')
# for item in test_json_list:
#     with open(os.path.join(dataset_root_path, 'test', item.replace(".json",".txt")), 'w') as f:
#         with open(os.path.join(dataset_root_path, 'test', item), 'r') as json_file:
#             json_data = json.load(json_file)
#             for shape in json_data['shapes']:
#                 #遍历shapes中的项，获取到points内容中两个点的中心点坐标
#                 x = (shape['points'][0][0] + shape['points'][1][0])/2
#                 y = (shape['points'][0][1] + shape['points'][1][1])/2
#                 #将中心点坐标写入txt文件
#                 f.write(str(x) + ' '+str(y)+'\n')