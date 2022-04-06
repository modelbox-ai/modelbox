#!/bin/bash
ls -lh release

os_name=$(sed -nr '1s/^NAME="(.*)"$/\1/gp' /etc/os-release)
if [ "$os_name" == "Ubuntu" ];then
    rm -f release/*.rpm
    postfix="*.deb"
elif [ "$os_name" == "openEuler" ];then
    postfix="*.rpm"
fi

if [ $(ls /usr/local|grep "cuda") ];then
    type="cuda"
elif [ $(ls /usr/local|grep "Ascend") ];then
    type="ascend"
fi

filecount=$(ls release | wc -l)
dpkgcount=$(ls release | egrep ${postfix} | wc -l)
artifacts_file=$(ls ${artifacts_path} | grep ${type}| wc -l)

if [ ${filecount} -ge 14 ] && [ ${dpkgcount} -ge 12 ] && [ ${artifacts_file} -eq 2 ]; then
    echo "compile success"
else
    echo "compile failed"
    exit 1
fi
