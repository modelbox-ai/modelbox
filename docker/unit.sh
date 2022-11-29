#!/system/bin/sh
cd /data/user/test/bin
LD_LIBRARY_PATH=../dep:../drivers \
LD_PRELOAD=libgmock.so:libgtest.so:libmodelbox.so:libsecurec.so \
HOME=/data/user/test \
PYTHONHOME=/data/user/test/python3.10 \
PYTHONPATH=/data/user/test/python3.10:/data/user/test/python3.10/lib-dynload:/data/user/test/python3.10/site-packages \
./unit --gtest_filter="-CryptoTest.AesEncryptPass:PopenTest.*"
