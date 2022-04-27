#!/bin/bash
ls -lh ./build/release
cd ./build
os_name=$(sed -nr '1s/^NAME="(.*)"$/\1/gp' /etc/os-release)
if [ "$os_name" == "Ubuntu" ];then
    rm -f release/*.rpm
    postfix="*.deb"
elif [ "$os_name" == "openEuler" ];then
    postfix="*.rpm"
fi

if [ $(ls release|grep "cuda"|wc -l) -eq 2 ];then
    type="cuda"
elif [ $(ls release|grep "ascend"|wc -l) -eq 2 ];then
    type="ascend"
fi

filecount=$(ls release | wc -l)
pkgcount=$(ls release | egrep "${postfix}" | wc -l)
artifacts_file=$(ls release | grep "${type}"| wc -l)

if [ ${filecount} -ge 14 ] && [ ${pkgcount} -ge 12 ] && [ ${artifacts_file} -eq 2 ]; then
    echo "compile success"
else
    echo "compile failed"
    exit 1
fi
