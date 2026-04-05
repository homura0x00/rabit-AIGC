# QAKGproject

## 1.运行环境

| 环境          | 属性     |
| ------------- | :------- |
| 操作系统      | window11 |
| python        | 3.8      |
| neo4j图数据库 | 5.2.0    |
|               |          |



## 2.知识图谱创建与服务

> 在命令行中，输入命令启动neo4j数据库

~~~powershell
neo4j.bat console
~~~

 

> 在QAsever目录下，进行知识图谱创建

~~~python
python3 CreateKnowGraph.py
~~~



## 3.启动医疗客服机器人

> 在QAsever目录下，执行命令启动服务

~~~python
python3 chatbot.py
~~~

