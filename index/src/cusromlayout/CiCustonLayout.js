import React,{Component}from 'react'
import Layout from "antd/es/layout";
const { Header, Content, Footer, Sider } = Layout;
class CiCustonLayout extends Component{
    constructor(props) {
        super(props);
        this.state={}

    }
    render() {
        return (
            <Layout style={{ padding: '24px 0', background: '#fff',width:700,margin:'0 auto' }}>
                <Content style={{ padding: '0 24px', minHeight: 780 }} overlay="">
                   <div>
                       Content
                   </div>
                </Content>
            </Layout>
        );
    }
}
export default CiCustonLayout;