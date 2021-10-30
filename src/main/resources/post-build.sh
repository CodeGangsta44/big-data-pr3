#!/bin/sh
export RESULTS_DIRECTORY=/home/roman_dovhopoliuk_big_data/results

rm -rf ./results
mkdir results

ssh -i /home/windows/.ssh/bigdata-cluster \
  roman_dovhopoliuk_big_data@$SPARK_IP \
  "rm -rf ${RESULTS_DIRECTORY}"

ssh -i /home/windows/.ssh/bigdata-cluster \
  roman_dovhopoliuk_big_data@$SPARK_IP \
  'hadoop fs -get /lab03/results /home/roman_dovhopoliuk_big_data'

scp -r -i /home/windows/.ssh/bigdata-cluster \
  roman_dovhopoliuk_big_data@$SPARK_IP:${RESULTS_DIRECTORY}/*.csv  \
  ./results

rm result.csv
echo "price,odometer,year,cluster" > result.csv
cat ./results/*csv >> result.csv

