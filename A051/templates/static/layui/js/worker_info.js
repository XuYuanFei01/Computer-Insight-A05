import '../layui.js';

let addButton=document.querySelector('#add-btn');
addButton.addEventListener('click',function () {
    layui.use('layer',function () {
        layer.alert('wuruiqi');
    })
})