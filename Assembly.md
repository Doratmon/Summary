## 分段管理
* 物理地址
	* 就是一个存储单元编号
	* 每个物理存储单元有一个20位编号（针对8086cpu）
	* 8086cpu物理地址寻址范围：00000H~FFFFFH
* 逻辑地址
	* 段基地址：段内偏移地址
	* 物理地址：将逻辑地址地址左移四位，加上偏移地址得到物理地址
* 段寄存器与逻辑段（8086有四个段寄存器）
	* CS：ip取得下一条要执行的指令
	* 用SS：SP指明堆栈段的起始地址
	* 用DS：EA存取数据段中的数据
	* 利用ES：EA存取附加段中的数据
* 没有指明段前缀时，一般在访问DS**(数据段)**
	* MOV AX,1000H——>MOV AX,DS:[1000H]
## 标志寄存器
* 分为状态标志与控制标志 

##  指令及寻址方式
* 指令分为操作码和操作数
	* 零操作数->指令格式中没有操作数或操作数**隐含**
	* 一操作数 ->隐含了一个操作数
	* 二操作数   一个为目的操作数 一个为源操作数
## 寻址方式
* 定义：指令中指明操作数存放位置的表达方式
* 类别
	* 立即数
	* 寄存器
	* 存储器