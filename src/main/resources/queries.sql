create
external table vehicles_sample
(
id string,
url string,
region string,
region_url string,
price string,
year string,
manufacturer string,
model string,
condition string,
cylinders string,
fuel string,
odometer string,
title_status string,
VIN string,
drive string,
size string,
type string,
paint_color string,
image_url string,
description string,
county string,
state string,
lat string,
long string,
posting_date string
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
STORED AS TEXTFILE
LOCATION '/lab03-sample/';



-- id,url,region,region_url,price,year,manufacturer,model,condition,cylinders,fuel,odometer,title_status,transmission,VIN,drive,size,type,paint_color,image_url,description,county,state,lat,long,posting_date
-- 7222695916,https://prescott.craigslist.org/cto/d/prescott-2010-ford-ranger/7222695916.html,prescott,https://prescott.craigslist.org,6000,,,,,,,,,,,,,,,,,,az,,,
-- 7218891961,https://fayar.craigslist.org/ctd/d/bentonville-2017-hyundai-elantra-se/7218891961.html,fayetteville,https://fayar.craigslist.org,11900,,,,,,,,,,,,,,,,,,ar,,,
-- 7221797935,https://keys.craigslist.org/cto/d/summerland-key-2005-excursion/7221797935.html,florida keys,https://keys.craigslist.org,21000,,,,,,,,,,,,,,,,,,fl,,,
-- 7222270760,https://worcester.craigslist.org/cto/d/west-brookfield-2002-honda-odyssey-ex/7222270760.html,worcester / central MA,https://worcester.craigslist.org,1500,,,,,,,,,,,,,,,,,,ma,,,
-- 7210384030,https://greensboro.craigslist.org/cto/d/trinity-1965-chevrolet-truck/7210384030.html,greensboro,https://greensboro.craigslist.org,4900,,,,,,,,,,,,,,,,,,nc,,,
-- 7222379453,https://hudsonvalley.craigslist.org/cto/d/westtown-2007-ford-150/7222379453.html,hudson valley,https://hudsonvalley.craigslist.org,1600,,,,,,,,,,,,,,,,,,ny,,,
-- 7221952215,https://hudsonvalley.craigslist.org/cto/d/westtown-silverado-2000/7221952215.html,hudson valley,https://hudsonvalley.craigslist.org,1000,,,,,,,,,,,,,,,,,,ny,,,
-- 7220195662,https://hudsonvalley.craigslist.org/cto/d/poughquag-2015-acura-rdx-warranty/7220195662.html,hudson valley,https://hudsonvalley.craigslist.org,15995,,,,,,,,,,,,,,,,,,ny,,,
-- 7209064557,https://medford.craigslist.org/cto/d/grants-pass-two-2002-bmw-tii/7209064557.html,medford-ashland,https://medford.craigslist.org,5000,,,,,,,,,,,,,,,,,,or,,,
