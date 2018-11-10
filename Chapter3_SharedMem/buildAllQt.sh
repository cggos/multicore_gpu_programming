#!/bin/bash 

for f in `/bin/ls *.pro`
do
   qmake $f; make
done