import decimal
import json
import re
import cv2 as cv
import numpy as np
import pandas as pd
from datetime import time, datetime, timedelta
from django.core import serializers
from django.forms import model_to_dict
from django.http import JsonResponse, HttpResponse, StreamingHttpResponse, HttpResponseNotFound
from django.shortcuts import render
# Create your views here.
from django.shortcuts import render,redirect
from django.views.decorators.csrf import csrf_exempt
from django.core.paginator import Paginator, PageNotAnInteger, EmptyPage
from django.conf import settings    # 获取 settings.py 里边配置的信息
from django.template import loader,Context
from django.utils import timezone
import os
import torchvision.models as modelss
from torch import nn
from .models import *
import pymysql
import torch
'''
这里都是服务访问是连接html的函数，
函数名字也就是url中http:127.0.0.1:8000/worksystem/xxx
url访问的正是函数控制调用的对应的html 
所有的html都在template下面
其中
login文件夹对应的是登录注册模块的html
layui文件夹对应的是管理系统的主要模块html
info文件夹管理的是admin资料与安全模块的html
'''


#打开注册界面
def signup(request):
    return render(request,'login/sign-up.html')


#打开登录界面
def login(request):
    return render(request,'login/login.html')


#打开主页
def index(request):
    u = request.session.get('login_user_name')
    n = request.session.get('login_name')
    print(u,n)
    if request.session.get('is_login', None):
        return render(request, 'layui/index.html', {'username': u,'name':n})
    else:
        return redirect('workersystem:login')


#打开监控区
def display(request):
    u = request.session.get('login_user_name')
    n = request.session.get('login_name')
    if request.session.get('is_login', None):
        return render(request, 'layui/video_paly_always.html',{'username': u,'name':n})
    else:
        return redirect('workersystem:login')


#打开工人信息栏目
def workers(request):
    u = request.session.get('login_user_name')
    n = request.session.get('login_name')
    print("zdddddd")
    if request.session.get('is_login', None):
        return render(request, 'layui/workerinfo.html',{'username': u,'name':n})
    else:
        return redirect('workersystem:login')


#打开工人信息详情界面
def moreworkerinfo(request):
    return render(request,'layui/more_info.html')


#打开违纪已读栏目
def ruleshaveread(request):
    u = request.session.get('login_user_name')
    n = request.session.get('login_name')
    if request.session.get('is_login', None):
        return render(request,'layui/rules_break_info_read.html',{'username': u,'name':n})
    else:
        return redirect('workersystem:login')


#打开违纪未读栏目
def rulesunread(request):
    u = request.session.get('login_user_name')
    n = request.session.get('login_name')
    if request.session.get('is_login', None):
        return render(request,'layui/rules_break_info.html',{'username': u,'name':n})
    else:
        return redirect('workersystem:login')


#添加工人窗口
def addworker(request):
    return render(request,'layui/add_worker.html')


# admin基本资料的界面
def userinfo(request):
    u = request.session.get('login_user_name')
    n = request.session.get('login_name')
    if request.session.get('is_login', None):
        return render(request, 'info/private_info.html', {'username': u, 'name': n})
    else:
        return redirect('workersystem:login')


# admin安全保护的界面
def usersafe(request):
    u = request.session.get('login_user_name')
    n = request.session.get('login_name')
    if request.session.get('is_login', None):
        return render(request, 'info/password_safe.html', {'username': u, 'name': n})
    else:
        return redirect('workersystem:login')


# 修改admin信息的窗口
def editinfo(request):
    return render(request, 'info/edit_info.html')


# 修改admin密码的窗口
def pwdsafe(request):
    return render(request, 'info/password_change.html')


'''
以下是函数实现部分
函数实现基本涉及到的是数据库操作，还有本地图片的增删改查操作
里面还涉及到json,img,二进制,字典元组的操作，
已经把很多辅助对接口的数据解析print全删除了
遇到问题可以使用这个方法来检测http请求的数据接收与发送
print(type(xxx))可以获取到对应的数据类型

关于数据库的操作有django自带的ORM查询，可以不通过sql语句
在1.0版本开发时为了早点完工就用了自己熟悉的手写方式,
具体可以参照getdata和getdata2对于工人信息的提取就采用了两种方法，
其中手写版的getdata舍去了
在登录的模块没有使用手写的sql查询，是为了防止sql注入问题。

关于本地保存的图片信息有以下说明：
在template下：

image/photo/pic_admin：
保存admin的照片，同时数据库中也保存了admin的二进制模式照片做了备份，
在sqlyog前端数据库界面管理工具下能正确解析成图片文件

****
图片命名：name_username.jpg
(name和username是admin数据库表的字段，代表admin的真实姓名和用户名)
****

image/photo/records:
这里存储的是关于records的记录，
算法方面对接时需要把cv摄像头获取的图片存到这里，

****
图片命名:工号_时间戳.jpg(时间戳去掉了-:还有空格符)
例如:008_20200101084531(时间戳原本为2020-01-01 08:45:31)
关于如何将这个提取出来的在deleterulesrecordlist中有代码辅助
****

静态文件加载需要在前端加上{% load static %}
静态资源文件都是存放在template/static下
css:存放自己写的css
js：存放自己写的js
font:存放美化字体
layui:存放layui的框架库包含layui相关的css，js，lib
base.html:在管理系统的html中采用了继承base.html的方式用来显示网页相同的部分，防止代码重用

A051的相关文件夹主要是全局设置，我基本已经配置好了，没有特殊需要尽量不修改django框架本身的settings.py和urls.py
workersystem下主要用到views编写后端代码，
每编写一个函数，一定要在workersystem的urls.py中添加path
该path的格式固定，也是前端连接后端发出请求的唯一url设置
其他文件基本不要动，
如果对于数据库的表有任何的修改,需要在models.py里面修改一下

数据库只有admin，records，workers跟我们相关其他框架自带

算法对接的要求主要是能讲record存到本地的static/image/records/下并按照规则命名，
同时将这个路径存储到records的src下，
src字符串按照我数据库里面写的那个格式存：
/static/image/records/008_20200101122120.jpg,这样才能够在web中访问到

'''

#添加管理员的功能实现函数
@csrf_exempt
def addadmin(request):
    name = request.POST['name']
    username = request.POST['username']
    password = request.POST['password']
    sex = request.POST['sex']
    a_file = request.FILES['form_data']
    photoname=str(name)+'_'+str(username)+'.jpg'
    filename = os.path.join(settings.MEDIA_ROOT, photoname)
    #admin的照片文件存到static/image/pic_admin
    with open(filename, 'wb') as f:
        img = a_file.file.read()
        f.write(img)
        f.close()
    if(sex=='男'):
        sex=0
    else:
        sex=1
    conn = pymysql.connect(host='localhost', user='root', password='123456', database='a05', port=3306)
    cursor = conn.cursor()
    sql = "insert into admin(name,username,password,photo,gender)values(%s,%s,%s,%s,%s)"
    cursor.execute(sql, (name, username, password, pymysql.Binary(img),sex))
    # 提交，不然无法保存新建或者修改的数据
    conn.commit()
    # 关闭游标
    cursor.close()
    # 关闭连接
    conn.close()
    return redirect('workersystem:login')

#登录逻辑实现函数
@csrf_exempt
def do_login(request):
    username = request.POST['username']
    password = request.POST['password']
    if (username == '' or password == ''):
        return redirect('workersystem:login')
    admin=Admin.objects.filter(username=username,password=password)
    if admin:
        ad = model_to_dict(admin[0])
        request.session['login_user_name'] = username
        request.session['login_name'] = ad['name']
        request.session['is_login'] = '1'
        request.session.set_expiry(0)
        return render(request, 'layui/index.html', {'username': request.session.get('login_user_name'),'name': request.session.get('login_name')})
    else:
        return HttpResponse("Your username and password didn't match.")

#添加工人信息实现函数
@csrf_exempt
def workersignup(request):
   # try:
    workerid = request.POST['id']
    name = request.POST['name']
    gender = request.POST['sex']
    a_file = request.FILES['photo']
    ects=a_file.name.split(".")[1]
    photoname=str(workerid)+'_'+str(name)+'.jpg'
    # admin的照片文件存到static/image/pic_admin
    filename = os.path.join(settings.MEDIA_ROOT, photoname)
    with open(filename, 'wb') as f:
        img = a_file.file.read()
        f.write(img)
        f.close()
    scale = 0.1
    fa=cv.imread(filename)
    img2 = cv.resize(fa, (int(fa.shape[1] * scale), int(fa.shape[0] * scale)),
                 interpolation=cv.INTER_LINEAR)
    face_cascade = cv.CascadeClassifier('E:/test\A051/templates\static/face_cascade\haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(img2, 1.1, 5)
    if len(faces) == 1:
        face = chopFace(img2, faces[0])
        # face是人脸区域缩放成(FACE_SIZE, FACE_SIZE)的正方形图像
        face = cv.resize(face, (256, 256), interpolation=cv.INTER_LINEAR)
    else:
        print("Warning: no face or more than 1 faces detected")
        face = chopCenter(img2)
    img_tensor = torch.from_numpy(face.astype(np.float32) / 255).permute(2, 0, 1).unsqueeze(0)
    model_extract_face_feature=modelss.resnet18(pretrained=True)
    model_extract_face_feature.fc = nn.Identity()
    model_extract_face_feature.eval()
    face_feature = model_extract_face_feature(img_tensor)
    nparr = face_feature.detach().numpy()
    face_feature = nparr[0]
    face_feature_bytes = face_feature.tostring()
    conn = pymysql.connect(host='localhost', user='root', password='123456', database='a05', port=3306)
    cursor = conn.cursor()
    if(gender=='男'):
        gender=0
    else:
        gender=1
    print(workerid)
    sql = "insert into workers(workerid,name,gender,photo,face_feature)values(%s,%s,%s,%s,%s)"
    # 执行SQL语句
    cursor.execute(sql, (workerid, name, gender, pymysql.Binary(img), face_feature_bytes))
    try:
       # 提交修改
        conn.commit()
    except:
        # 发生错误时回滚
        print("s")
        conn.rollback()
        # 关闭游标
        cursor.close()
        # 关闭连接
        conn.close()
        return HttpResponse("error", status=404)
    # 关闭游标
    cursor.close()
    # 关闭连接
    conn.close()
    return HttpResponse("ok",status=200)


#暂时已经废掉的获取工人信息的表格
def getdata(request):
    try:
        conn = pymysql.connect(host='localhost', user='root', password='123456', database='a05', port=3306)
        cur = conn.cursor()
        sql = "SELECT workerid,name,gender from workers"
        # execute可执行数据库查询select和命令insert，delete，update三种命令(这三种命令需要commit()或rollback())
        cur.execute(sql)
        # 5.获取数据
        # fetchall遍历execute执行的结果集。取execute执行后放在缓冲区的数据，遍历结果，返回数据。
        # 返回的数据类型是元组类型，每个条数据元素为元组类型:(('第一条数据的字段1的值','第一条数据的字段2的值',...,'第一条数据的字段N的值'),(第二条数据),...,(第N条数据))
        data = cur.fetchall()
        cur.close()
        # 7.关闭connection
        conn.close()
        jsonData = []
        jsonData['count']= len(data)
        # 循环读取元组数据
        # 将元组数据转换为列表类型，每个条数据元素为字典类型:[{'字段1':'字段1的值','字段2':'字段2的值',...,'字段N:字段N的值'},{第二条数据},...,{第N条数据}]
        for row in data:
            result = {}
            result['id'] = row[0]
            result['name'] = row[1]
            if(row[2]==0):
                result['sex'] = "男"
            else:
                result['sex'] = "女"
            jsonData.append(result)
        print(u'转换为列表字典的原始数据：', jsonData)
    except:
        print('MySQL connect fail...')
    else:
        jsondatar = json.dumps(jsonData, ensure_ascii=False)
# 使用json.dumps将数据转换为json格式，json.dumps方法默认会输出成这种格式"\u5377\u76ae\u6298\u6263"，加ensure_ascii=False，则能够防止中文乱码。
# JSON采用完全独立于语言的文本格式，事实上大部分现代计算机语言都以某种形式支持它们。这使得一种数据格式在同样基于这些结构的编程语言之间交换成为可能。
# json.dumps()是将原始数据转为json（其中单引号会变为双引号），而json.loads()是将json转为原始数据。
    return HttpResponse(jsondatar,content_type='application/json; charset=utf-8')
    # 去除首尾的中括号
    # return JsonResponse(jsondatar, safe=False, json_dumps_params={'ensure_ascii': False})

#获取工人信息的表格实现
def getdata2(request):
    page=request.GET['page']
    pageSize=request.GET['limit']
    #print(page)
    #print(pageSize)
    master_list = Workers.objects.order_by('workerid')
    # 查询整个Master
    #print(master_list)
    ret = Workers.objects.all()
    lenth=len(ret)
    json_list = []
    k=1
    for i in ret:
        result = {}
        if (k >= (1+(int(page)-1)*int(pageSize)) and k<=int(pageSize)*int(page)):
            json_dict = model_to_dict(i)
            result['workerid'] = json_dict['workerid']
            result['name'] = json_dict['name']
            if (json_dict['gender'] == 0):
                result['gender'] = "男"
            else:
                result['gender'] = "女"
            result['count']=lenth
            json_list.append(result)
        k=k+1
    #     print(k)
    # print(json_list)
    jsondatar = json.dumps(json_list, ensure_ascii=False)
    # 循环读取元组数据
    # 将元组数据转换为列表类型，每个条数据元素为字典类型:[{'字段1':'字段1的值','字段2':'字段2的值',...,'字段N:字段N的值'},{第二条数据},...,{第N条数据}]
    return HttpResponse(jsondatar, content_type='application/json; charset=utf-8')

#删除单个员工
@csrf_exempt
def deleteworker(request):
    wokerid=request.POST['workerid']
    conn = pymysql.connect(host='localhost', user='root', password='123456', database='a05', port=3306)
    cursor = conn.cursor()
    sql = "DELETE FROM workers WHERE workerid = %s " % wokerid
    cursor.execute(sql)
    try:
        # 提交修改
        conn.commit()
    except:
        # 发生错误时回滚
        conn.rollback()
        # 关闭游标
        cursor.close()
        # 关闭连接
        conn.close()
        return HttpResponse("error", status=404)
    # 关闭游标
    cursor.close()
    # 关闭连接
    conn.close()
    return HttpResponse("ok",status=200)

#工人信息的批量删除
@csrf_exempt
def deleteworkerlist(request):
    request_data = request.body#json提取
    request_dict = json.loads(request_data.decode('utf-8'))#json转字典
    conn = pymysql.connect(host='localhost', user='root', password='123456', database='a05', port=3306)
    cursor = conn.cursor()
    sql = "DELETE FROM workers WHERE workerid IN ("
    for i in range(len(request_dict)):
        if (i == 0):
            sql=sql+'\''+str(request_dict[i])+'\''
        else:
            sql=sql+','+'\''+str(request_dict[i])+'\''
    sql+=')'
    print(sql)
    cursor.execute(sql)
    try:
        # 提交修改
        conn.commit()
    except:
        # 发生错误时回滚
        conn.rollback()
        # 关闭游标
        cursor.close()
        # 关闭连接
        conn.close()
        return HttpResponse("error", status=404)
    # 关闭游标
    cursor.close()
    # 关闭连接
    conn.close()
    return HttpResponse("ok",status=200)

#工人信息详情获取
@csrf_exempt
def personget(request):
        workerid=request.GET['id']
        print(workerid)
        conn = pymysql.connect(host='localhost', user='root', password='123456', database='a05', port=3306)
        cursor = conn.cursor()
        sql = "SELECT * FROM records WHERE most_close_worker_id=%s AND isread=1 ORDER BY record_time DESC" %workerid
        cursor.execute(sql)
        # 提交，不然无法保存新建或者修改的数据
        conn.commit()
        data = cursor.fetchall()
        cursor.close()
        # 7.关闭connection
        conn.close()
        jsonData = []
        for row in data:
            result = {}
            result['time'] = str(row[1])
            result['possible'] =str(float(row[2])*100)+'%'
            result['src']=str(row[5])
            jsonData.append(result)
        jsondatar = json.dumps(jsonData, ensure_ascii=False)
    # 使用json.dumps将数据转换为json格式，json.dumps方法默认会输出成这种格式"\u5377\u76ae\u6298\u6263"，加ensure_ascii=False，则能够防止中文乱码。
    # JSON采用完全独立于语言的文本格式，事实上大部分现代计算机语言都以某种形式支持它们。这使得一种数据格式在同样基于这些结构的编程语言之间交换成为可能。
    # json.dumps()是将原始数据转为json（其中单引号会变为双引号），而json.loads()是将json转为原始数据。
        return HttpResponse(jsondatar, content_type='application/json; charset=utf-8')

#工人详细信息中的违纪信息记录提取
@csrf_exempt
def personruluesdelete(request):
    request_data = request.body  # json提取
    request_dict = json.loads(request_data.decode('utf-8'))# json转字典
    workerid = request_dict['workerid']
    record_time = request_dict['record_time']
    src = request_dict['src']
    src = src[7:]
    filepath = settings.STATICFILES_DIRS[0] + src
    os.remove(filepath)
    conn = pymysql.connect(host='localhost', user='root', password='123456', database='a05', port=3306)
    cursor = conn.cursor()
    sql = "DELETE FROM records WHERE most_close_worker_id = %s AND record_time = %s" % (('\''+workerid+'\''),('\''+str(record_time)+'\''))
    cursor.execute(sql)
    # 提交，不然无法保存新建或者修改的数据
    conn.commit()
    data = cursor.fetchall()
    cursor.close()
    # 7.关闭connection
    conn.close()
    return HttpResponse("ok",status=200)

#未读的违纪信息表格提取
def rulesunreaddata(request):
        page = request.GET['page']
        pageSize = request.GET['limit']
        conn = pymysql.connect(host='localhost', user='root', password='123456', database='a05', port=3306)
        cur = conn.cursor()
        sql = "SELECT * from records where isread=0 ORDER BY record_time DESC"
        cur.execute(sql)
        data = cur.fetchall()
        lenth=len(data)
        cur.close()
        # 7.关闭connection
        conn.close()
        jsonData = []
        k = 1
        for d in data:
            result = {}
            if (k >= (1 + (int(page) - 1) * int(pageSize)) and k <= int(pageSize) * int(page)):
                result['time'] = str(d[1])
                print(result['time'])
                result['possible'] = str(float(d[2]) * 100) + '%'
                result['id'] = str(d[3])
                result['name'] = str(d[4])
                result['src'] = str(d[5])
                result['count'] = lenth
                jsonData.append(result)
            k = k + 1
        jsondatar = json.dumps(jsonData, ensure_ascii=False)
        return HttpResponse(jsondatar, content_type='application/json; charset=utf-8')

#对未读信息进行批量已读
@csrf_exempt
def rulescomfirmreadlist(request):
    print("确认已读")
    request_data = request.body  # json提取
    print(request_data)
    request_dict = json.loads(request_data.decode('utf-8'))
    print(request_dict)
    workerid = request_dict['id']
    time = request_dict['time']
    conn = pymysql.connect(host='localhost', user='root', password='123456', database='a05', port=3306)
    cursor = conn.cursor()
    for i in range(len(workerid)):
        print(i)
        sql = "UPDATE records SET isread=1 WHERE most_close_worker_id ="
        sql2 = "AND record_time="
        sql = sql + '\'' + str(workerid[i]) + '\''
        sql2 = sql2 + '\'' + str(time[i]) + '\''
        sql = sql + sql2 + "AND isread=0"
        cursor.execute(sql)
    try:
        # 提交修改
        conn.commit()
    except:
        # 发生错误时回滚
        conn.rollback()
        # 关闭游标
        cursor.close()
        # 关闭连接
        conn.close()
        return HttpResponse("error", status=404)
    # 关闭游标
    cursor.close()
    # 关闭连接
    conn.close()
    return HttpResponse("ok", status=200)


#已读信息的表格提取
def ruleshavareaddata(request):
    page = request.GET['page']
    pageSize = request.GET['limit']
    conn = pymysql.connect(host='localhost', user='root', password='123456', database='a05', port=3306)
    cur = conn.cursor()
    sql = "SELECT * from records where isread=1 ORDER BY record_time DESC"
    # execute可执行数据库查询select和命令insert，delete，update三种命令(这三种命令需要commit()或rollback())
    cur.execute(sql)
    # 5.获取数据
    # fetchall遍历execute执行的结果集。取execute执行后放在缓冲区的数据，遍历结果，返回数据。
    # 返回的数据类型是元组类型，每个条数据元素为元组类型:(('第一条数据的字段1的值','第一条数据的字段2的值',...,'第一条数据的字段N的值'),(第二条数据),...,(第N条数据))
    data = cur.fetchall()
    lenth = len(data)
    print(lenth)
    cur.close()
    # 7.关闭connection
    conn.close()
    jsonData = []
    k = 1
    for d in data:
        result = {}
        if (k >= (1 + (int(page) - 1) * int(pageSize)) and k <= int(pageSize) * int(page)):
            result['time'] = str(d[1])
            print(result['time'])
            result['possible'] = str(float(d[2]) * 100) + '%'
            result['id'] = str(d[3])
            result['name'] = str(d[4])
            result['src'] = str(d[5])
            result['count'] = lenth
            jsonData.append(result)
        k = k + 1
    jsondatar = json.dumps(jsonData, ensure_ascii=False)
    return HttpResponse(jsondatar, content_type='application/json; charset=utf-8')

#对单个的未读信息进行已读
@csrf_exempt
def rulescomfirmread(request):
    print(1111111)
    request_data = request.body  # json提取
    print(request_data)
    request_dict = json.loads(request_data.decode('utf-8'))
    print(request_dict)
    workerid = request_dict['workerid']
    time=request_dict['time']
    conn = pymysql.connect(host='localhost', user='root', password='123456', database='a05', port=3306)
    cursor = conn.cursor()
    sql = "UPDATE records SET isread=1 WHERE most_close_worker_id = %s AND record_time=%s AND isread=0" % (('\''+workerid+'\''),('\''+str(time)+'\''))
    cursor.execute(sql)
    try:
        # 提交修改
        conn.commit()
    except:
        # 发生错误时回滚
        conn.rollback()
        # 关闭游标
        cursor.close()
        # 关闭连接
        conn.close()
        return HttpResponse("error", status=404)
    # 关闭游标
    cursor.close()
    # 关闭连接
    conn.close()
    return HttpResponse("ok", status=200)

#删除已读的单个信息
@csrf_exempt
def deleterulesrecord(request):
    request_data = request.body  # json提取
    request_dict = json.loads(request_data.decode('utf-8'))
    workerid = request_dict['workerid']
    time = request_dict['time']
    dt=pd.to_datetime(time)
    #删除本地照片下,static/image/records/workerid_年份字符串(例如008_20201229133256)时间戳去掉符号2020-12-29 13:32:56
    dti=dt.strftime("%Y%m%d%H%M%S")
    path=settings.STATICFILES_DIRS[0]+'\\image\\photo\\records\\'+workerid+'_'+dti+'.jpg'
    os.remove(path)
    conn = pymysql.connect(host='localhost', user='root', password='123456', database='a05', port=3306)
    cursor = conn.cursor()
    sql = "DELETE FROM records WHERE most_close_worker_id = %s AND record_time = %s" % (('\''+workerid+'\''),('\''+str(time)+'\''))
    try:
        # 执行SQL语句
        cursor.execute(sql)
        # 提交修改
        conn.commit()
    except:
        # 发生错误时回滚
        conn.rollback()
        # 关闭游标
        cursor.close()
        # 关闭连接
        conn.close()
        return HttpResponse("error")
    # 关闭游标
    cursor.close()
    # 关闭连接
    conn.close()
    return HttpResponse("ok", status=200)

#批量删除已读信息
@csrf_exempt
def deleterulesrecordlist(request):
    request_data = request.body  # json提取
    request_dict = json.loads(request_data.decode('utf-8'))
    id = request_dict['id']
    time = request_dict['Time']
    #用了pandas的方式来批量处理本地照片时间戳格式转换来获取工人的record图片名字
    dtime=pd.to_datetime(time)
    dtime = [i.strftime("%Y%m%d%H%M%S") for i in dtime]  # 修改时间格式为
    path = [settings.STATICFILES_DIRS[0] + '\\image\\photo\\records\\' + i + '_' + t + '.jpg' for i,t in zip(id,dtime)]
    conn = pymysql.connect(host='localhost', user='root', password='123456', database='a05', port=3306)
    cursor = conn.cursor()
    for i in range(len(id)):
        sql = "DELETE FROM records WHERE most_close_worker_id  ="
        sql2 = "AND record_time ="
        sql = sql + '\'' + str(id[i]) + '\''
        sql2 = sql2 + '\'' + str(time[i]) + '\''
        sql = sql+sql2
        print(sql)
        os.remove(path[i])
        cursor.execute(sql)
    try:
        # 提交修改
        conn.commit()
    except:
        # 发生错误时回滚
        conn.rollback()
        # 关闭游标
        cursor.close()
        # 关闭连接
        conn.close()
        return HttpResponse("error", status=404)
    # 关闭游标
    cursor.close()
    # 关闭连接
    conn.close()
    return HttpResponse("ok", status=200)

#首页曲线图近七天数据获取
def getday(request):
    conn = pymysql.connect(host='localhost', user='root', password='123456', database='a05', port=3306)
    cursor = conn.cursor()
    sql = "SELECT * FROM records e WHERE DATE_SUB(CURDATE(), INTERVAL 6 DAY) <= DATE(record_time)"
    cursor.execute(sql)
    # 提交，不然无法保存新建或者修改的数据
    conn.commit()
    data = cursor.fetchall()
    cursor.close()
    # 7.关闭connection
    conn.close()
    #之后可以改成np或者pd来操作
    counts = [0, 0, 0, 0, 0, 0, 0]
    d6 = datetime.now()
    d0 = d6 - timedelta(days=6)
    d1 = d6 - timedelta(days=5)
    d2 = d6 - timedelta(days=4)
    d3 = d6 - timedelta(days=3)
    d4 = d6 - timedelta(days=2)
    d5 = d6 - timedelta(days=1)
    print(d5)
    for day in data:
        if (day[1] <= d0):
            counts[0] = counts[0] + 1
        elif (day[1] < d1):
            counts[1] = counts[1] + 1
        elif (day[1] < d2):
            counts[2] = counts[2] + 1
        elif (day[1] < d3):
            counts[3] = counts[3] + 1
        elif (day[1] < d4):
            counts[4] = counts[4] + 1
        elif (day[1] < d5):
            counts[5] = counts[5] + 1
        else:
            counts[6] = counts[6] + 1
    print(counts)
    jsondatar = json.dumps(counts, ensure_ascii=False)
    print(jsondatar)
    return HttpResponse(jsondatar, content_type='application/json; charset=utf-8')

#首页月份违纪统计记录
def getmonth(request):
    conn = pymysql.connect(host='localhost', user='root', password='123456', database='a05', port=3306)
    cursor = conn.cursor()
    sql = "SELECT * FROM records e WHERE DATE_SUB(CURDATE(), INTERVAL 6 MONTH) <= DATE(record_time)"
    cursor.execute(sql)
    # 提交，不然无法保存新建或者修改的数据
    conn.commit()
    data = cursor.fetchall()
    cursor.close()
    # 7.关闭connection
    conn.close()
    counts = [0, 0, 0, 0, 0, 0]
    d6 = datetime.now()
    d0 = d6 - timedelta(days=180)
    d1 = d6 - timedelta(days=150)
    d2 = d6 - timedelta(days=120)
    d3 = d6 - timedelta(days=90)
    d4 = d6 - timedelta(days=60)
    d5 = d6 - timedelta(days=30)
    for day in data:
        if (day[1] >= d0 and day[1] < d1):
            counts[0] = counts[0] + 1
        elif (day[1] >= d1 and day[1] < d2):
            counts[1] = counts[1] + 1
        elif (day[1] >= d2 and day[1] < d3):
            counts[2] = counts[2] + 1
        elif (day[1] >= d3 and day[1] < d4):
            counts[3] = counts[3] + 1
        elif (day[1] >= d4 and day[1] < d5):
            counts[4] = counts[4] + 1
        else:
            counts[5] = counts[5] + 1
    jsondatar = json.dumps(counts, ensure_ascii=False)
    return HttpResponse(jsondatar, content_type='application/json; charset=utf-8')

#首页获取员工的数量
def workercounts(request):
    conn = pymysql.connect(host='localhost', user='root', password='123456', database='a05', port=3306)
    cursor = conn.cursor()
    sql = "SELECT * FROM workers"
    cursor.execute(sql)
    # 提交，不然无法保存新建或者修改的数据
    conn.commit()
    data = cursor.fetchall()
    cursor.close()
    # 7.关闭connection
    conn.close()
    return HttpResponse(len(data))

#退出按钮
def logout(request):
    request.session.flush()
    return redirect('workersystem:login')

#获取登录的session，用来防止横向越界，并传递username到前端用来切换跟账号相关的信息头像
def getSession(request):
    data={}
    if (request.session.get('is_login') == 1):
        data={
            'name':request.session.get('name'),
            'username':request.session.get('username')
        }
        return JsonResponse(data)
    else:
        return JsonResponse(data)



#修改密码的逻辑实现函数
@csrf_exempt
def changepwd(request):
    u = request.session.get('login_user_name')
    n = request.session.get('login_name')
    org=request.POST['orign']
    new=request.POST['new']
    conn = pymysql.connect(host='localhost', user='root', password='123456', database='a05', port=3306)
    cursor = conn.cursor()
    sql = "update admin set password= '%s' where name='%s' and username='%s' and password='%s' " %(new,n,u,org)
    print(sql)
    cursor.execute(sql)
    try:
        # 提交修改
        conn.commit()
    except:
        # 发生错误时回滚
        conn.rollback()
        # 关闭游标
        cursor.close()
        # 关闭连接
        conn.close()
        return HttpResponse("error", status=404)
    # 关闭游标
    cursor.close()
    # 关闭连接
    conn.close()
    return HttpResponse("ok", status=200)

#绑定邮箱的实现函数
def changemail(request):
    u = request.session.get('login_user_name')
    n = request.session.get('login_name')
    email=request.GET['em']
    conn = pymysql.connect(host='localhost', user='root', password='123456', database='a05', port=3306)
    cursor = conn.cursor()
    sql = "update admin set email= '%s' where name='%s' and username='%s' " % (email, n, u)
    cursor.execute(sql)
    try:
        # 提交修改
        conn.commit()
    except:
        # 发生错误时回滚
        conn.rollback()
        # 关闭游标
        cursor.close()
        # 关闭连接
        conn.close()
        return HttpResponse("error", status=404)
    # 关闭游标
    cursor.close()
    # 关闭连接
    conn.close()
    return HttpResponse("ok", status=200)

#修改admin基本信息的函数
@csrf_exempt
def updateinfo(request):
    u = request.session.get('login_user_name')
    n = request.session.get('login_name')
    name=request.POST['truename']
    username=request.POST['username']
    gender=request.POST['sex']
    text=request.POST['desc']
    print(text)
    if (gender == '男'):
        gender = 0
    else:
        gender = 1
    conn = pymysql.connect(host='localhost', user='root', password='123456', database='a05', port=3306)
    cursor = conn.cursor()
    sql = "update admin set username='%s',name='%s', gender='%s',text='%s'  where name='%s' and username='%s'  " %(username,name,gender,text,n,u)
    print(sql)
    cursor.execute(sql)
    try:
        # 提交修改
        conn.commit()
    except:
        # 发生错误时回滚
        conn.rollback()
        # 关闭游标
        cursor.close()
        # 关闭连接
        conn.close()
        return HttpResponse("error", status=404)
    # 关闭游标
    cursor.close()
    # 关闭连接
    conn.close()
    return HttpResponse("ok", status=200)

#发送admin中的性别和个人介绍到前端
def getadmin(request):
    u = request.session.get('login_user_name')
    n = request.session.get('login_name')
    conn = pymysql.connect(host='localhost', user='root', password='123456', database='a05', port=3306)
    cursor = conn.cursor()
    sql = "select gender,text from admin where name='%s' and username='%s' " %(n,u)
    print(sql)
    cursor.execute(sql)
    try:
        # 提交修改
        conn.commit()
        data = cursor.fetchall()
        print(data)
        print(type(data))
        jsonData = []
        for row in data:
            result = {}
            if (row[0] == 0):
                result['sex'] = "男"
            else:
                result['sex'] = "女"
            result['text'] = row[1]
            jsonData.append(result)
            break
        jsondatar = json.dumps(jsonData, ensure_ascii=False)
    except:
        # 发生错误时回滚
        conn.rollback()
        # 关闭游标
        cursor.close()
        # 关闭连接
        conn.close()
        return HttpResponse("error", status=404)
    #关闭游标
    cursor.close()
    # 关闭连接
    conn.close()
    return HttpResponse(jsondatar, status=200)

#密码强度等级计算
def getlv(request):
    u = request.session.get('login_user_name')
    n = request.session.get('login_name')
    conn = pymysql.connect(host='localhost', user='root', password='123456', database='a05', port=3306)
    cursor = conn.cursor()
    sql = "select password,email from admin where name='%s' and username='%s' " %(n,u)
    cursor.execute(sql)
    try:
        conn.commit()
        data = cursor.fetchall()
        jsonData = []
        for row in data:
            result={}
            s=row[0]
            ls=0
            if(checkLower(s)|checkSymbol(s)):
                ls=ls+1
            if(checkUpper(s)):
                ls=ls+1
            if(checkNum(s)):
                ls=ls+1
            if (len(s) >= 10):
                ls=ls+1
            elif(len(s) < 10):
                ls=1
            result['count']=ls
            result['email'] = row[1]
            jsonData.append(result)
            break
        jsondatar = json.dumps(jsonData, ensure_ascii=False)
    except:
        # 发生错误时回滚
        conn.rollback()
        # 关闭游标
        cursor.close()
        # 关闭连接
        conn.close()
        return HttpResponse("error", status=404)
    # 关闭游标
    cursor.close()
    # 关闭连接
    conn.close()
    return HttpResponse(jsondatar, status=200)

#以下是密码强度的检测相关函数
def checkUpper(data):
    upper = re.compile('[A-Z]+')
    match = upper.findall(data)
    if match:
        return True
    else:
        return False
def checkLower(data):
    lower = re.compile('[a-z]+')
    match = lower.findall(data)
    if match:
        return True
    else:
        return False
def checkNum(data):
    num = re.compile('[0-9]+')
    match = num.findall(data)
    if match:
        return True
    else:
        return False
def checkSymbol(data):
    symbol = re.compile('([^a-zA-Z0-9])+')
    match = symbol.findall(data)
    if match:
        return True
    else:
        return False


@csrf_exempt
def insert_video(request):
    u =request.session.get('login_user_name')
    time_now = timezone.localtime(timezone.now())
    time_local=time_now.strftime("%Y-%m-%d %H:%M:%S")
    user=str(u)
    print(user)
    conn = pymysql.connect(host='localhost', user='root', password='123456', database='a05', port=3306)
    cursor = conn.cursor()
    sql = "insert into video(start_time,adminster,machine)values(%s,%s,%s)"
    # 执行SQL语句
    cursor.execute(sql, (time_local,u,"树莓派"))
    try:
        conn.commit()
    except:
        # 发生错误时回滚
        conn.rollback()
        # 关闭游标
        cursor.close()
        # 关闭连接
        conn.close()
        # 关闭游标
    data = cursor.fetchone()
    print(data)
    cursor.close()
    # 关闭连接
    conn.close()
    return HttpResponse("ok",status=200)


@csrf_exempt
def get_video_time(request):
    conn = pymysql.connect(host='localhost', user='root', password='123456', database='a05', port=3306)
    cursor = conn.cursor()
    sql = "SELECT start_time FROM video ORDER BY id DESC LIMIT 1"
    cursor.execute(sql)
    try:
        conn.commit()
    except:
        # 发生错误时回滚
        conn.rollback()
        # 关闭游标
        cursor.close()
        # 关闭连接
        conn.close()
        # 关闭游标
    data = cursor.fetchone()
    cursor.close()
    # 关闭连接
    conn.close()
    return HttpResponse(data)

@csrf_exempt
def get_video_times(request):
    conn = pymysql.connect(host='localhost', user='root', password='123456', database='a05', port=3306)
    cursor = conn.cursor()
    sql = "SELECT * FROM video "
    cursor.execute(sql)
    try:
        conn.commit()
    except:
        # 发生错误时回滚
        conn.rollback()
        # 关闭游标
        cursor.close()
        # 关闭连接
        conn.close()
        # 关闭游标
    data = cursor.fetchall()
    cursor.close()
    # 关闭连接
    conn.close()
    return HttpResponse(len(data))


@csrf_exempt
def get_user(request):
    u =request.session.get('login_user_name')
    return HttpResponse(u)


def chopFace(img, pos, expand=True):
    (x, y, w, h) = pos
    if not (img.ndim == 3 and img.shape[2] == 3):
        return img
    centerY = y + h//2
    centerX = x + w//2
    if expand:
        halfSideLength = int((max(w, h)//2) * 1.1)
        # 防止放大的区域超出图片边界
        possibleConstraints = (halfSideLength, centerX, centerY, img.shape[0]-centerY, img.shape[1]-centerX)
        halfSideLength = min(possibleConstraints)
    else:
        halfSideLength = (max(w, h) // 2)
    print(img.shape)
    print(halfSideLength)
    return img[centerY-halfSideLength:centerY+halfSideLength,
           centerX-halfSideLength:centerX+halfSideLength, :]

def chopCenter(img):
    if not (img.ndim == 3 and img.shape[2] == 3):
        return img
    halfSideLength = min(img.shape[0], img.shape[1])//2 - 1
    centerY = img.shape[0]//2
    centerX = img.shape[1]//2
    return img[centerY-halfSideLength:centerY+halfSideLength,
           centerX-halfSideLength:centerX+halfSideLength]