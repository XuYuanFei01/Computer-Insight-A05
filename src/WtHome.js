import { Layout, Menu, Icon } from 'antd';
import React,{Component} from "react";
import * as config from "@babel/core";
import {Router,Route,Link,Switch} from 'react-router-dom'
import CiCheck from "./check/CiCheck";
import './WtHome.css';
import CiWorkList from "./worklist/CiWorkList";
import CiPersonSetting from "./menbersetting/CiPersonSetting";
import CiCustonLayout from "./cusromlayout/CiCustonLayout";
import SubMenu from "antd/es/menu/SubMenu";
const { Header, Content, Footer, Sider } = Layout;

class CiNormalLayout extends Component{

    render(){
      let items=[];
        items.push(<Route path="/home/task-check" component={CiCheck}/>);
        items.push(<Route path="/home/task-list" component={CiWorkList}/>);
        items.push(<Route path="/home/member-setting" component={CiPersonSetting}/>);
        items.push(<Route path="/home/custom-layout" component={CiCustonLayout}/>);
        return (
            <Layout className="mainLayout" style={{ padding: '24px 0', background: '#fff' }}>
                <Content style={{ padding: '0 24px', minHeight: 780 }} overlay="">
                    <Switch>
                        {items}
                    </Switch>
                </Content>
            </Layout>
        );
    }
}
class WtHome extends  Component{
        constructor(props){
            super(props);
            this.state={
                collapsed: false,
                mimg:false,
            }
        }
        renderSubMenu(){
            let items=[];
            items.push(
                <Menu.Item key="task-list">
                    <Icon type="edit" />
                    <Link to="/home/task-list">未处理信息</Link>
                </Menu.Item>
            );
            items.push(
                <Menu.Item key="task-check">
                    <Icon type="team" />
                    <Link to="/home/task-check">查看违纪人员</Link>
                </Menu.Item>
            );
            items.push(
                <Menu.Item key="member-setting"  >
                    <Icon type="user" />
                    <Link to="/home/member-setting">个人设置</Link>
                 </Menu.Item>
            );
            return (
                <Menu
                    className="wt-left-menu"
                    theme="light"
                    mode="inline"
                    // selectedKeys={[secondLevel]}
                >
                    {items}
                </Menu>
            );
        }
    toggle = () => {
        this.setState({
            collapsed: !this.state.collapsed,
            mimg:!this.state.mimg,
        });
    };
    onLogout = () => {

    };

    render(){
        return (
            <div>
                <header className="header">
                    <div className="logo" style={{float:'left',color:'#808080',height:62}}>
                        <span style={{fontSize:20,color:'white'}}>违纪员工管理系统<span className="student-version" > V1.0.0</span></span>
                    </div>

                    <div style={{float:'right',color:'#808080',height:62}}>
                        <span style={{padding:'0 20px',color:'white'}}>名字</span>
                        <a href="javascript:location.href='http://laravel.ykh:81/A05/index/public/login.html' " style={{padding:'0 20px',color:'white'}} onClick={this.onLogout}>退出</a>
                    </div>
                </header>

                <main className="main">
                    <div className="menu">
                        {this.renderSubMenu()}
                    </div>

                    <div className="content">
                            <Route path="/home" render={() => (<CiNormalLayout />)}/>
                    </div>
                </main>
            </div>
        );
    }
}
export default WtHome;