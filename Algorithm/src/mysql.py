# coding=utf-8
"""
https://blog.csdn.net/RunffyCSDN/article/details/81486880
"""
import pymysql
import numpy as np
import datetime
import cv2
FEATURE_SIZE = 512
"""
create database A05 charset=utf8;

use A05

create table workers(
id int auto_increment primary key,
name varchar(10) not null,
gender bit default 0,
photo blob,
face_feature blob
);

create table records(
id int auto_increment primary key,
time datetime,
wear_hat_confidence DECIMAL(4,4),
most_close_worker_id int
);
"""


class MysqlOperator(object):
    def __init__(self, ip='localhost', user='root', password='123456', db='A05'):
        self.database = pymysql.connect(host=ip,
                                        port=3306,
                                        user=user, password=password,
                                        db=db, charset='utf8')
        self.cur = self.database.cursor()

    def search_record_by_id(self, id):
        """
        搜索某个工人所有报警记录
        :param id: 
        :return: 
        """
        sql = "select * from records where most_close_worker_id = %s"
        self.cur.execute(sql, (id,))
        results = self.cur.fetchall()
        print(results)
        print(type(results[0][2]))


    def _hasThisTable(self, table_name):
        '''
        判断是否存在此表
        :param table_name:表名 
        :return: True  or  False
        '''
        sql = "show tables;"
        self.cur.execute(sql)
        results = self.cur.fetchall()

        for r in results:
            if r[0] == table_name:
                return True
        else:
            return False

    def _hasThisId(self, table_name, dateID):
        '''
        判断在此表中是否已经有此主键
        :param table_name: 表名
        :param dateID: 主键值
        :return: True  or  False
        '''
        sql = "select dates from " + table_name + ";"
        self.cur.execute(sql)
        ids = self.cur.fetchall()
        for i in ids:
            if i[0] == dateID:
                return True
        else:
            return False

    def insertRecord(self, wear_hat_confidence, most_close_worker_id,datetime,name,photo):
        """
        插入一条危险警报记录
        :param wear_het_confidence: 
        :param most_close_worker_id: 
        :return: 
        """
        print("incerting record..")
        dt1 = datetime.strftime("%Y-%m-%d %H:%M:%S")
        dt2 = datetime.strftime("%Y%m%d%H%M%S")
        filename=str(most_close_worker_id)+'_'+str(dt2)+'.jpg'
        path='E:/test/A051/templates/static/image/photo/records/'+filename
        cv2.imwrite(path, photo)
        print(path)
        img='/static/image/photo/records/'+filename
        sql = "insert into records(record_time, wear_hat_confidence, most_close_worker_id,name,photo,isread) values(%s, %s, %s,%s,%s,%s)"
        self.cur.execute(sql, (dt1, wear_hat_confidence, most_close_worker_id,name,img,0))
        self.database.commit()

    def insertWorker(self,workerid, name, gender, photo, face_feature):
        """
        增加一条工人信息
        :param name:
        :param gender:
        :param nparr:
        :return:
        """
        photo_bytes = photo.tostring()
        face_feature_bytes = face_feature.tostring()
        sql = "insert into workers(workerid,name, gender, photo, face_feature) values(%s, %s, %s, %s, %s)"

        self.cur.execute(sql, (workerid,name, gender, photo_bytes, face_feature_bytes))
        self.database.commit()

    def readData(self):
        """
        读取所有的工人的人脸特征向量
        :return: 
        """
        sql = "select * from workers"
        self.cur.execute(sql)
        # 执行sql语句
        results = self.cur.fetchall()
        print(len(results))
        face_features = np.zeros((len(results), FEATURE_SIZE))
        # 获取所有结果集
        for i, row in enumerate(results):
            # row[0]为默认自增主键
            numArr = np.fromstring(string=row[5], dtype=np.float32)
            face_features[i] = numArr
        return face_features

    def __del__(self):
        """
        临走之前记得关灯关电关空调，还有关闭数据库资源
        :return: 
        """
        if hasattr(self, 'database'):
            self.database.close()

if __name__ == '__main__':
    mysql = MysqlOperator()
    mysql.search_record_by_id(2)

