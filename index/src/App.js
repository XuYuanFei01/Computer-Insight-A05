import React,{Component} from 'react';
//import {Provider} from 'react-redux'
import logo from './logo.svg';

import './App.css';
import{LocaleProvider} from'antd'
import locale from 'antd/lib/locale-provider/zh_CN';
import moment from 'moment';
import 'moment/locale/zh-cn';
import WtHome from "./WtHome";
moment.locale('zh-cn');

export default class App extends Component {
  render() {
    return (
        <provider>
          <LocaleProvider locale={locale}>
             <div>
                 <WtHome/>
             </div>
          </LocaleProvider>
        </provider>
    );
  }
}
