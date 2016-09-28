START=`date +%s%N`;

#sleep 2;
#THIS IS RUN ON TIANHE-2
yhrun ./kmeanst --num_clusters 4 --input datann.txt --output result.txt --init_method random 
#yhrun ./kmeanst --num_clusters 100 --input datann.txt --output result.txt 

END=`date +%s%N`;
time=$((END-START));
time=`expr $time / 1000000`
#time=$(time)
echo TIME:

echo $time
