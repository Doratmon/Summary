# JSON格式
## json.dump()与json.load()
* json.dump()用于接收两个对象一个是要写入的python数据，一个可用于存储数据的文件对象(`with open("data.txt",'w') as f: 中的f就是可用于存储数据的文件对象`)，用于将
python数据结构写入.json文件中
* json.load()只有一个对象就是json文件对象
## json.dumps()与json.loads()
* json.dumps()用于将python数据结构转换为json编码的字符串
* json.loads()用于将json编码的字符串转换为python数据结构
## 区别：
* 多一个s处理字符串(String),少一个s为处理文件
* [详见链接](https://www.cnblogs.com/everfight/p/json_file.html)
-------------
