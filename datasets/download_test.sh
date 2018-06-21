#!/bin/bash
cd ./datasets

wget -O test.tar.gz https://umich.box.com/shared/static/5q39kgt2ftkfy352rr2lmwuzrhqjxwvy.gz

tar -xvzf test.tar.gz

rm test.tar.gz
mv test/*npy .

cd ../

