#!/bin/bash
# 
# 将项目打包压缩（用于正式打包）
# Author: alex
# Created Time: 2020年05月10日 星期日
cd ../
name="pytorch-CycleGAN-and-pix2pix"
if [ ! -d "$name" ]; then
    echo "$PWD: 当前目录错误."
fi
version=
if [ $# = 1 ]; then 
    version="-$1"
fi

# 删除旧的打包文件
date_str=`date -I`
filename="$name-${date_str//-/}$version".tar.gz
if [ -f "$filename" ]; then
    rm -f "$filename"
fi

if which zip; then
    echo 'zip is ok.'
else
    echo 'zip is not installed!'
    exit 1
fi

# 第一次打包时，记得把模型文件也打包在一起
# 正式发布时，记得过滤src目录中的内容
tar -zcvf "$filename" \
    --exclude ".git" \
    --exclude ".*" \
    --exclude ".swp" \
    --exclude ".ipynb" \
    --exclude ".tar.gz" \
    --exclude "__pycache__" \
    --exclude "zip.sh" \
    "$name"

echo 'scp to ibbd@245:~/notebook/work/git'
scp "$filename" ibbd@192.168.80.245:~/notebook/work/git

date
echo ""
echo ""
