* git bash显示中文问题 右键options->text(Zh_CH UTF-8)
* 命令行输入 git config --global core.quotepath false



### git diff用法

* git diff ： 对比工作区(未 git add)和暂存区(git add 之后)
* git diff --cached: 对比暂存区(git add 之后)和版本库(git commit 之后)
* git diff HEAD:  对比工作区(未 git add)和版本库(git commit 之后)

### git add

* git add -u：将文件的修改、文件的删除，添加到暂存区。
  git add .：将文件的修改，文件的新建，添加到暂存区。
  git add -A：将文件的修改，文件的删除，文件的新建，添加到暂存区。

* git add -A  保存所有的修改

  git add .   保存新的添加和修改，但是不包括删除

  git add -u  保存修改和删除，但是不包括新建文件。