#!/bin/bash
name="60_200"
r="1_reward"
exp="NoExp"
file="r12_VT_inc"
cd $name
cd $r
cd $exp
cd $file
python a3_1.py >> a3_1.txt
python a3_2.py >> a3_2.txt    
python a3_3.py >> a3_3.txt