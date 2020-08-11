### VMware中虚拟机已启动了NAT模式，运行ipconfig/ifconfig仍然无法查看到ip地址，无法上网

* 编辑->虚拟网络编辑器->点击还原默认设置
* 在服务中确保VMware NAT Service已启动
* _遇到问题记得查看相应服务有没有启动是一个好习惯_

### GCC

* gcc包含的c/c++编译器：gcc、cc、c++、g++

  gcc和cc是一样的，c++和g++是一样的，一般c程序用gcc编译，c++程序用g++编译

* 其实cc就是一个软件链接到gcc，只不过cc是UNIX中常用的编译工具，在linux中用的是gcc，有一些在UNIX中写好的程序放在linux中要指定命令cc，所以就将cc指定为gcc

  * 一般的cc就是一个软件链接，可以先用whereis cc查看所在路径，再用ls -l命令查看详细信息，可以看到一个箭头（-->）指向一个可执行文件。



### Office经验总结

* word中大小写互相切换`shift+F3`

### Linux（CentOS7）无法上网，NAT模式

* ifconfig查看网卡信息，有一块类似ens33的网卡

* 再vim /etc/sysconfig/network-scripts/ifcfg-ens33

  ```
  
  BOOTPROTO=static  #static为静态ip，dhcp为动态分配
  
  ONBOOT=yes  #这里如果为no的话就改为yes，表示网卡设备自动启动
  
  ------下面是设置static需要配置的-----
  GATEWAY=192.168.10.2  #这里的网关地址就是第二步获取到的那个网关地址
  IPADDR=192.168.10.150  #配置ip，不能和网关冲突
  NETMASK=255.255.255.0#子网掩码
  DNS1=202.96.128.86#dns服务器1，填写你所在的网络可用的dns服务器地址即可
  DNS2=223.5.5.5#dns服器2
  ```

### U盘退出被占用问题

* 计算机右键----管理-------事件查看器-----自定义视图------管理事件------双击最近发生的事件---------查看哪个进程阻止了U盘退出，将其kill即可
* 这个针对产生警告事件的情况

