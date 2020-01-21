<?php
    header('Content-type:text/html;charset=utf-8');
    //连接认证
    $link=mysqli_connect('localhost:3306','root','123456') or die('数据库失败');
    //选择数据库
    mysqli_query($link,'use a05');

