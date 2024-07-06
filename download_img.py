import paramiko
import os
from stat import S_ISDIR as isdir
import pandas as pd
import json

def download_img(sftp,remote_dir):
    imgs = ["CXR3222_IM-1522"]
    for img in imgs:
        if not os.path.exists('/Users/woqingdoua/Downloads/human_evaluation/iu_xray/' + img):
            os.makedirs('/Users/woqingdoua/Downloads/human_evaluation/iu_xray/' + img)
        for remote_file_name in sftp.listdir(remote_dir+'/images/'+img):
            sub_remote = os.path.join(remote_dir+'/images/'+img, remote_file_name)
            sftp.get(sub_remote, '/Users/woqingdoua/Downloads/human_evaluation/iu_xray/' + img + '/'+remote_file_name)

'''
def download_img(sftp,remote_dir):
    file = '/Users/woqingdoua/Downloads/radiology_report_human_evaluation.csv'
    data = pd.read_csv(file)
    images_id = data['images_id'].values
    for img in images_id:
        try:
            if '.jpg' not in img:
                if not os.path.exists('/Users/woqingdoua/Downloads/human_evaluation/iu_xray/' + img):
                    os.makedirs('/Users/woqingdoua/Downloads/human_evaluation/iu_xray/' + img)
                for remote_file_name in sftp.listdir(remote_dir+'/images/'+img):
                    sub_remote = os.path.join(remote_dir+'/images/'+img, remote_file_name)
                    sftp.get(sub_remote, '/Users/woqingdoua/Downloads/human_evaluation/iu_xray/' + img + '/'+remote_file_name)

            else:
                local_dir_name = '/Users/woqingdoua/Downloads/human_evaluation/mimic'
                sftp.get(img, local_dir_name+'/'+os.path.basename(img))
        except:
            pass
'''


if __name__ == "__main__":
    """程序主入口"""
    # 服务器连接信息
    host_name = '141.225.8.108'
    user_name = 'ywu10'
    password = 'memphis123'
    port = 22
    # 远程文件路径（需要绝对路径）
    remote_dir = '/home/ywu10/Documents/R2Gen/data/iu_xray'
    # 本地文件存放路径（绝对路径或者相对路径都可以）
    local_dir = '/Users/woqingdoua/Downloads/human_evaluation'

    # 连接远程服务器
    t = paramiko.Transport((host_name, port))
    t.connect(username=user_name, password=password)
    sftp = paramiko.SFTPClient.from_transport(t)

    # 远程文件开始下载
    download_img(sftp, remote_dir)

    # 关闭连接
    t.close()


'''
file1 = '/home/ywu10/Documents/R2Gen/data/iu_xray/annotation.json'
file2 = '/home/ywu10/Documents/R2Gen/data/mimic_cxr/annotation.json'
data = pd.read_csv('/home/ywu10/Documents/r2genbaseline/radiology_report_human_evaluation.csv')
iu_xray_img = data['images_id'][:50]
mimic_img = data['images_id'][50:100]
truth = []
file1 = json.loads(open(file1, 'r').read())['test']
file2 = json.loads(open(file2, 'r').read())['test']

for i in iu_xray_img:
    for j in file1:
        if j['id'] == i:
            truth.append(j['report'])

for i in mimic_img :
    for j in file2:
        if j['image_path'][0] in i:
            truth.append(j['report'].replace('\n',''))

data = pd.DataFrame({'truth':truth})
data.to_csv('truth.csv')
'''

'''
data = pd.read_csv('/Users/woqingdoua/Downloads/radiology_report_human_evaluation.csv')
images_id = data['images_id'][50:]
iii = []
for i in images_id:
    try:
        iii.append(os.path.basename(i))
    except:
        pass

dd = pd.DataFrame({'images':iii})
dd.to_csv('images_id.csv')
'''