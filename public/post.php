<?php
    include_once '16database.php';
    $name=$_POST['name'];
    $username=$_POST['username'];
    $password=$_POST['password'];
    //echo $name." ".$username." ".$password;
    //$form_description = $_POST['form_description'];
    $form_data_name = $_FILES['form_data']['name'];
    $form_data_size = $_FILES['form_data']['size'];
    $form_data_type = $_FILES['form_data']['type'];
    $form_data = $_FILES['form_data']['tmp_name'];
    $data = addslashes(fread(fopen($form_data, "r"), filesize($form_data)));
    //echo "mysqlPicture=".$data;
    $result = $link->query("INSERT INTO admin (name,username,password,photo)
         VALUES ('{$name}','{$username}','{$password}','{$data}')");
    if ($result) {
        echo "<script>alert('图片已存储到数据库')</script>";
        header('refresh:0.1;url=login.html');
    } else {
        echo "<script>alert('请求失败,请重试')</script>";
        echo "<script>window.history.go(-1)</script>";
    }



