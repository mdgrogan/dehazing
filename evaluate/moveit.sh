#!/bin/sh

#for it in {51..100}
#do
#    cp /home/grogan/Storage/CSCE633_Project_data/its/clear/${it}.png "img/clahe/$it.png"
#done

for it in {51..100}
do
    cp /home/grogan/Storage/CSCE633_Project_data/its/hazy/${it}_1_*.*.png "img/clahe/$it.png"
done
