################
#Author: Chris Wilson
#Title: County Level Aggregation Code
#Date: 11/10/20
################
library(tidyverse)
library(RCurl)
library(reshape2)

link_table <- read.csv('/home/ec2-user/covid_data/HRR_FIPS.csv')
git_link  <- getURL("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv")
data_set <- read.csv(text = git_link)

joined_set<-merge(x=link_table,y=data_set,by="FIPS",all.x=TRUE)
for(row in 1:nrow(joined_set)){
  temp_row<-joined_set[row,]
  temp_row[,20:dim(temp_row)[2]]<-as.numeric(temp_row[,6])*temp_row[,20:dim(temp_row)[2]]
  joined_set[row,]<-temp_row
}

by_HRR<-subset.data.frame(joined_set,select=-c(FIPS,HRR,County_Name,pop10,
                                               area_join_pcent,iso2,iso3,
                                               code3,Admin2,Province_State,
                                               Country_Region,c_lon,c_lat,State,
                                               Lat,Long_,Combined_Key,UID))
by_HRR<-by_HRR %>% group_by(HRRName)
by_HRR<-by_HRR %>% summarize_all(list(sum))
#pivot
data_long <- 
  melt(by_HRR,
       id.vars=c("HRRName"),
       measure.vars = dput(grep("[0-9]",names(by_HRR),value = T)),
       variable.name = "Date",
       value.name = "Confirm_count")

#correct date values
data_long$new_date<-as.Date(lubridate::mdy(str_remove(data_long[,2],'X')))

#rename and save
colnames(data_long)<-c('HRRName','drop','Confirm_count','Date')
data_long$drop<-NULL
data_long<-data_long[rev(order(data_long$Date)),]
#save
write.csv(data_long,'/home/ec2-user/rona_folder/covid_data/HRR_county_agg.csv')

link_table <- read.csv('/home/ec2-user/covid_data/sta6a_fips_CatchmentArea_80.csv')
colnames(link_table)<-c('Sta6a','FIPS')
link_table<-link_table[,1:2]
joined_set<-merge(x=link_table,y=data_set,by="FIPS",all.x=TRUE)

by_sta6a <- subset.data.frame(joined_set,select=-c(FIPS,UID,iso2,iso3,code3,
                                                   Admin2,Province_State,
                                                   Country_Region,
                                                   Lat,Long_,Combined_Key))

by_sta6a <- by_sta6a %>% group_by(Sta6a)
by_sta6a <- by_sta6a %>% summarize_all(list(sum))

data_long <- 
  melt(by_sta6a,
       id.vars=c("Sta6a"),
       measure.vars = dput(grep("[0-9]",names(by_sta6a),value = T)),
       variable.name = "Date",
       value.name = "Confirm_count")

#correct date values
data_long$new_date<-as.Date(lubridate::mdy(str_remove(data_long[,2],'X')))
colnames(data_long)<-c('Sta6a','drop','Confirm_count','Date')

data_long<-data_long[complete.cases(data_long[ ,4]),]
data_long$drop<-NULL


data_long<-data_long[rev(order(data_long$Date)),]
#save
write.csv(data_long,'/home/ec2-user/covid_data/Sta6a_county_agg.csv')

link_table <- read.csv('/home/ec2-user/covid_data/sta6a_fips_CatchmentArea_RespDisease_v2.csv')
colnames(link_table)<-c('Sta6a','nope','FIPS')
link_table<-link_table[,1:3]
joined_set<-merge(x=link_table,y=data_set,by="FIPS",all.x=TRUE)


by_ARD<-subset.data.frame(joined_set,select=-c(FIPS,UID,nope,UID,
                                               iso2,iso3,code3,Admin2,
                                               Province_State,Country_Region,
                                               Lat,Long_,Combined_Key))
by_ARD<-by_ARD %>% group_by(Sta6a)
by_ARD<-by_ARD %>% summarize_all(list(sum))

#pivot
data_long <- 
  melt(by_ARD,
       id.vars=c("Sta6a"),
       measure.vars = dput(grep("[0-9]",names(by_ARD),value = T)),
       variable.name = "Date",
       value.name = "Confirm_count")

#correct date values
data_long$new_date<-as.Date(lubridate::mdy(str_remove(data_long[,2],'X')))
colnames(data_long)<-c('Sta6a','drop','Confirm_count','Date')

data_long<-data_long[complete.cases(data_long[ ,4]),]
data_long$drop<-NULL
#save
data_long<-data_long[rev(order(data_long$Date)),]
write.csv(data_long,'/home/ec2-user/covid_data/ARDs_county_aggv2.csv')#save

by_fips<-subset.data.frame(data_set,select=-c(UID,iso2,iso3,code3,Admin2,
                                              Province_State,Country_Region,
                                              Lat,Long_,Combined_Key))
by_fips<-by_fips %>% group_by(FIPS)
by_fips<-by_fips %>% summarize_all(list(sum))

#pivot
data_long <- 
  melt(by_fips,
       id.vars=c("FIPS"),
       measure.vars = dput(grep("[0-9]",names(by_fips),value = T)),
       variable.name = "Date",
       value.name = "Confirm_count")

#correct date values
data_long$new_date<-as.Date(lubridate::mdy(str_remove(data_long[,2],'X')))
colnames(data_long)<-c('FIPS','drop','Confirm_count','Date')

data_long<-data_long[complete.cases(data_long[ ,4]),]
data_long$drop<-NULL
#save
data_long<-data_long[rev(order(data_long$Date)),]
write.csv(data_long,'/home/ec2-user/covid_data/FIPS_county_agg.csv')#save

