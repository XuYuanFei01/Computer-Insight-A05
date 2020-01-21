<?php
    include_once '16database.php';
    //var_dump($_POST);
    $username=$_POST['username'];
    $password=$_POST['password'];
    if(($username=='')||($password==''))
    {
        header('refresh:0.1;url=login.html');
        echo "<script>alert('改用户名或密码不能为空')</script>";
        exit;
    }
    $result = mysqli_query($link,"select * from admin where username='{$username}' and password='{$password}' ");
    if($result->num_rows > 0)
    {
        echo  "<script>setTimeout(\"javascript:location.href='http://localhost:3000/home/task-list/{$username}'\", 3000)</script>";
    }
    else {
        echo "<script>alert('登录失败')</script>";
    }

