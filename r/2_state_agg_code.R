#################################################################################
## FileName: COVID19_Tom.R
## Author: Peter Chen
## Date Create: 3/26/20
## Modified: 11/20/20; Chris W.
#################################################################################
### Set local library

library(data.table)
library(janitor)
library(Rcpp)
library(Hmisc)
library(pander)
library(plyr)
library(httr)
library(readr)
library(rmarkdown)
library(knitr)


getwd()
#setwd("C:/Users/VHAECHChenH/temp/COVID19/")

# Import data from github
death_url<-"https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"
COVID19_death <-
  data.table(read_csv(url(death_url)))

confirm_url<-"https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
COVID19_confirm <-
  data.table(read_csv(url(confirm_url)))

#pivot wide to long
COVID19_death_long <- 
  melt(COVID19_death,
       id.vars=c("Province/State", "Country/Region", "Lat", "Long"),
       measure.vars = dput(grep("[0-9]",names(COVID19_death),value = T)),
       variable.name = "Date",
       value.name = "Death_count")

COVID19_confirm_long <- 
  melt(COVID19_confirm,
       id.vars=c("Province/State", "Country/Region", "Lat", "Long"),
       measure.vars = dput(grep("[0-9]",names(COVID19_confirm),value = T)),
       variable.name = "Date",
       value.name = "Confirm_count")

# Merge
COVID19_raw<-
  merge(x=COVID19_confirm_long,y=COVID19_death_long,
        by=c("Province/State", "Country/Region","Date"),
        all.x=T)
colnames(COVID19_raw)<-c('Province/State','Country/Region','Date','Lat','Long','Confirm_count','drop1','drop2','Death_count')
COVID19_raw$drop1<-NULL
COVID19_raw$drop2<-NULL
# Data management
dput(names(COVID19_raw))
COVID19_raw[,Date:=as.Date(Date,format = "%m/%d/%y")]

colnames(COVID19_raw) <- gsub(" |/", '_', colnames(COVID19_raw))

# Create flag for cruise
COVID19_raw[,cruise:=0]
COVID19_raw[Province_State%in% c("From Diamond Princess","Diamond Princess", "Grand Princess"),
            cruise:=1]
COVID19_raw<-COVID19_raw[cruise==0]

# Rename Korea
COVID19_raw[Country_Region=="Korea, South", Country_Region:="South Korea"]

# Check all country and states
unique(COVID19_raw[,Country_Region])
unique(COVID19_raw[,Province_State])

# Check the highest confirm counts among country
COVID19_raw[order(Confirm_count),tail(.SD, 1L), by=Country_Region][order(-Confirm_count), .SD[1:10]]

# Country analysis: 
country_interest<- c("Austria", "Belgium", "Czech Republic", " Finland", "Greece", "Italy",  
                     "Spain", "Germany", "France", "Israel" , "Netherlands" , "Norway" ,"Sweden", 
                     "Switzerland", "United Kingdom", "US", "South Korea")
COVID19_country<-
  COVID19_raw[Country_Region %in% country_interest]

# Just the country level data
COVID19_tom<-COVID19_country[,.(Total_confirm=Confirm_count, Total_death=Death_count), by=c("Date", "Country_Region", "Province_State")]
COVID19_tom[Country_Region=="France",.N,by=.(Country_Region, Province_State)]
COVID19_tom[Country_Region== "US", Country_Region:="United States of America"]
COVID19_tom[, new_confirm:= Total_confirm-shift(Total_confirm,n=1, type="lag"), by=c("Country_Region", "Province_State")]
COVID19_tom[, new_death:= Total_death-shift(Total_death,n=1, type="lag"), by=c("Country_Region", "Province_State")]

# Import World country pop data
World_pop <-
  data.table(read_csv("/home/ec2-user/covid_data/population_country.csv"))

# Rename Korea
World_pop[Location=="Republic of Korea", Location:="South Korea"]

# merge with COVID19_tom
COVID19_tom2<-merge(COVID19_tom,World_pop[,.(Location, PopTotal, PopDensity)], by.x="Country_Region", by.y="Location", all.x=T)

COVID19_tom2[Country_Region=="South Korea"]
## output RData object
write.csv(COVID19_tom2, file=paste0('COVID19_country','.csv'))

##############################    US state report   ############################
# Add daily report file
daily_url<-paste0("https://covidtracking.com/api/states/daily.csv")
df <-
  data.table(read_csv(url(daily_url)))

df[,Date:=as.Date(gsub('(.{4})(.{2})(.*)', "\\1-\\2-\\3", as.character(date)))]

COVID19_us_state<- 
  df[,.(Date, state, Confirm_count=positive, Death_count=death, Total_negative=negative, Total_hospitalized=hospitalized, Total_tested=total,
        New_confirm=positiveIncrease, New_death=deathIncrease, New_hospitalized=hospitalizedIncrease)]

COVID19_us_state[state=="AK", Province_State:="Alaska"]
COVID19_us_state[state=="AL", Province_State:="Alabama"]
COVID19_us_state[state=="AZ", Province_State:="Arizona"]
COVID19_us_state[state=="AR", Province_State:="Arkansas"]
COVID19_us_state[state=="CA", Province_State:="California"]
COVID19_us_state[state=="CO", Province_State:="Colorado"]
COVID19_us_state[state=="CT", Province_State:="Connecticut"]
COVID19_us_state[state=="DE", Province_State:="Delaware"]
COVID19_us_state[state=="FL", Province_State:="Florida"]
COVID19_us_state[state=="GA", Province_State:="Georgia"]
COVID19_us_state[state=="HI", Province_State:="Hawaii"]
COVID19_us_state[state=="ID", Province_State:="Idaho"]
COVID19_us_state[state=="IL", Province_State:="Illinois"]
COVID19_us_state[state=="IN", Province_State:="Indiana"]
COVID19_us_state[state=="IA", Province_State:="Iowa"]
COVID19_us_state[state=="KS", Province_State:="Kansas"]
COVID19_us_state[state=="KY", Province_State:="Kentucky"]
COVID19_us_state[state=="LA", Province_State:="Louisiana"]
COVID19_us_state[state=="ME", Province_State:="Maine"]
COVID19_us_state[state=="MD", Province_State:="Maryland"]
COVID19_us_state[state=="MA", Province_State:="Massachusetts"]
COVID19_us_state[state=="MI", Province_State:="Michigan"]
COVID19_us_state[state=="MN", Province_State:="Minnesota"]
COVID19_us_state[state=="MS", Province_State:="Mississippi"]
COVID19_us_state[state=="MO", Province_State:="Missouri"]
COVID19_us_state[state=="MT", Province_State:="Montana"]
COVID19_us_state[state=="NV", Province_State:="Nevada"]
COVID19_us_state[state=="NE", Province_State:="Nebraska"]
COVID19_us_state[state=="NH", Province_State:="New Hampshire"]
COVID19_us_state[state=="NJ", Province_State:="New Jersey"]
COVID19_us_state[state=="NM", Province_State:="New Mexico"]
COVID19_us_state[state=="NY", Province_State:="New York"]
COVID19_us_state[state=="NC", Province_State:="North Carolina"]
COVID19_us_state[state=="ND", Province_State:="North Dakota"]
COVID19_us_state[state=="OH", Province_State:="Ohio"]
COVID19_us_state[state=="OK", Province_State:="Oklahoma"]
COVID19_us_state[state=="OR", Province_State:="Oregon"]
COVID19_us_state[state=="PA", Province_State:="Pennsylvania"]
COVID19_us_state[state=="RI", Province_State:="Rhode Island"]
COVID19_us_state[state=="SC", Province_State:="South Carolina"]
COVID19_us_state[state=="SD", Province_State:="South Dakota"]
COVID19_us_state[state=="TN", Province_State:="Tennessee"]
COVID19_us_state[state=="TX", Province_State:="Texas"]
COVID19_us_state[state=="UT", Province_State:="Utah"]
COVID19_us_state[state=="VT", Province_State:="Vermont"]
COVID19_us_state[state=="VA", Province_State:="Virginia"]
COVID19_us_state[state=="WA", Province_State:="Washington"]
COVID19_us_state[state=="WV", Province_State:="West Virginia"]
COVID19_us_state[state=="WI", Province_State:="Wisconsin"]
COVID19_us_state[state=="WY", Province_State:="Wyoming"]
COVID19_us_state[state=="DC", Province_State:="District of Columbia"]

# Import US state pop data
US_pop <-
  data.table(read_csv("/home/ec2-user/covid_data/nst-est2019-alldata.csv"))
US_pop<- US_pop[,c("NAME","POPESTIMATE2019")]

# merge with COVID19_us_state
COVID19_US<-merge(COVID19_us_state,US_pop, by.x="Province_State", by.y="NAME", all.x=T)

## output RData object
write.csv(COVID19_US, file=paste0('/home/ec2-user/covid_data/COVID19_USTracking','.csv'))

#################   USJH Report  #########################

# Add daily report file
# 01-22 
get_early_report<- function(dates){
  
  listofdfs <- list() #Create a list in which you intend to save your df's.
  
  for (i in 1:length(dates)){ #Loop through the numbers of dates instead of the dates
    
    #Use dates[i] instead of i to get the date
    
    daily_url<-paste0("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/", dates[i], ".csv")
    df <-
      data.table(read_csv(url(daily_url)))
    colnames(df) <- gsub(" |/", '_', colnames(df))
    df[,Date:=as.Date(substr(Last_Update, start = 1, stop = 9), "%m/%d/%Y")] 
    listofdfs[[i]] <- df[,.(Province_State, Country_Region, Date, Confirmed, Deaths, Recovered)] # save your dataframes into the list
  }
  
  return(listofdfs) #Return the list of dataframes.
}

dateids<- as.character(format(as.Date("2020-01-22"), "%m-%d-%Y"))
COVID19_0122_list<- get_early_report(dates = dateids)
# unlist as data.table
COVID19_0122<-do.call(rbind, lapply(COVID19_0122_list, data.table, stringsAsFactors=FALSE))
COVID19_0122_unique<-unique(COVID19_0122)


# 01-23 ~ 01-30 (date formate)
get_early_report<- function(dates){
  
  listofdfs <- list() #Create a list in which you intend to save your df's.
  
  for (i in 1:length(dates)){ #Loop through the numbers of dates instead of the dates
    
    #Use dates[i] instead of i to get the date
    
    daily_url<-paste0("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/", dates[i], ".csv")
    df <-
      data.table(read_csv(url(daily_url)))
    colnames(df) <- gsub(" |/", '_', colnames(df))
    df[,Date:=as.Date(Last_Update, "%m/%d/%y")] 
    listofdfs[[i]] <- df[,.(Province_State, Country_Region, Date, Confirmed, Deaths, Recovered)] # save your dataframes into the list
  }
  
  return(listofdfs) #Return the list of dataframes.
}

dateids<- as.character(format(as.Date(seq(as.Date("2020-01-23"), as.Date("2020-01-30"), by=1)), "%m-%d-%Y"))
COVID19_early_list<- get_early_report(dates = dateids)
# unlist as data.table
COVID19_early<-do.call(rbind, lapply(COVID19_early_list, data.table, stringsAsFactors=FALSE))
COVID19_early_unique<-unique(COVID19_early)

# 01-31~02-01 
get_early_report<- function(dates){
  
  listofdfs <- list() #Create a list in which you intend to save your df's.
  
  for (i in 1:length(dates)){ #Loop through the numbers of dates instead of the dates
    
    #Use dates[i] instead of i to get the date
    
    daily_url<-paste0("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/", dates[i], ".csv")
    df <-
      data.table(read_csv(url(daily_url)))
    colnames(df) <- gsub(" |/", '_', colnames(df))
    df[,Date:=as.Date(substr(Last_Update, start = 1, stop = 9), "%m/%d/%Y")] 
    listofdfs[[i]] <- df[,.(Province_State, Country_Region, Date, Confirmed, Deaths, Recovered)] # save your dataframes into the list
  }
  
  return(listofdfs) #Return the list of dataframes.
}

dateids<-  as.character(format(as.Date(seq(as.Date("2020-01-31"), as.Date("2020-02-01"), by=1)), "%m-%d-%Y"))
COVID19_0201_list<- get_early_report(dates = dateids)
# unlist as data.table
COVID19_0201<-do.call(rbind, lapply(COVID19_0201_list, data.table, stringsAsFactors=FALSE))
COVID19_0201_unique<-unique(COVID19_0201)

# 02-02 ~ 03-21 (date format)
get_early_report<- function(dates){
  
  listofdfs <- list() #Create a list in which you intend to save your df's.
  
  for (i in 1:length(dates)){ #Loop through the numbers of dates instead of the dates
    
    #Use dates[i] instead of i to get the date
    
    daily_url<-paste0("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/", dates[i], ".csv")
    df <-
      data.table(read_csv(url(daily_url)))
    colnames(df) <- gsub(" |/", '_', colnames(df))
    df[,Date:=as.Date(as.POSIXct(Last_Update, 'GMT'))]  
    listofdfs[[i]] <- df[,.(Province_State, Country_Region, Date, Confirmed, Deaths, Recovered)] # save your dataframes into the list
  }
  
  return(listofdfs) #Return the list of dataframes.
}

dateids<- as.character(format(as.Date(seq(as.Date("2020-02-02"), as.Date("2020-03-21"), by=1)), "%m-%d-%Y"))
COVID19_mid_list<- get_early_report(dates = dateids)
# unlist as data.table
COVID19_mid<-do.call(rbind, lapply(COVID19_mid_list, data.table, stringsAsFactors=FALSE))
COVID19_mid_unique<-unique(COVID19_mid)
COVID19_mid_unique[Province_State=="Arizona" & Confirmed ==13,Date:=as.Date("2020-03-15")]

# 03-22
daily_url<-paste0("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/03-22-2020.csv")
df <-
  data.table(read_csv(url(daily_url)))
keep_variables<- c("Province_State", "Country_Region", "Last_Update", "Confirmed", "Deaths", "Recovered")
COVID19_0322<- df[,..keep_variables]
COVID19_0322<-COVID19_0322[,Date:=as.Date(Last_Update, "%m/%d/%y")][,.(Confirmed, Deaths, Recovered), by=.(Province_State, Country_Region, Date)]
COVID19_0322_unique<-unique(COVID19_0322)



# 03-23 ~ 03-28
get_recent_report<- function(dates){
  
  listofdfs <- list() #Create a list in which you intend to save your df's.
  
  for (i in 1:length(dates)){ #Loop through the numbers of dates instead of the dates
    
    #Use dates[i] instead of i to get the date
    
    daily_url<-paste0("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/", dates[i], ".csv")
    df <-
      data.table(read_csv(url(daily_url)))
    keep_variables<- c("Province_State", "Country_Region", "Last_Update", "Confirmed", "Deaths", "Recovered")
    COVID19_by_date<- df[,..keep_variables]
    COVID19_by_date[,Date:=as.Date(Last_Update, "%m/%d/%Y")] 
    listofdfs[[i]] <- COVID19_by_date[,.(Confirmed, Deaths, Recovered), by=.(Province_State, Country_Region, Date)] # save your dataframes into the list
  }
  
  return(listofdfs) #Return the list of dataframes.
}

dateids<- as.character(format(as.Date(seq(as.Date("2020-03-23"), as.Date("2020-03-27"), by=1)), "%m-%d-%Y"))
COVID19_0323_list<- get_recent_report(dates = dateids)
# unlist as data.table
COVID19_0323<-do.call(rbind, lapply(COVID19_0323_list, data.table, stringsAsFactors=FALSE))
COVID19_0323_unique<-unique(COVID19_0323)


# 03-28 -03-30 (3/28/20 23:08)
get_early_report<- function(dates){
  
  listofdfs <- list() #Create a list in which you intend to save your df's.
  
  for (i in 1:length(dates)){ #Loop through the numbers of dates instead of the dates
    
    #Use dates[i] instead of i to get the date
    
    daily_url<-paste0("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/", dates[i], ".csv")
    df <-
      data.table(read_csv(url(daily_url)))
    colnames(df) <- gsub(" |/", '_', colnames(df))
    df[,Date:=as.Date(Last_Update, "%m/%d/%y")] 
    listofdfs[[i]] <- df[,.(Province_State, Country_Region, Date, Confirmed, Deaths, Recovered)] # save your dataframes into the list
  }
  
  return(listofdfs) #Return the list of dataframes.
}

dateids<- c("03-28-2020", "03-29-2020", "03-30-2020", "04-02-2020", "04-04-2020") 
COVID19_0329_list<- get_early_report(dates = dateids)
# unlist as data.table
COVID19_0329<-do.call(rbind, lapply(COVID19_0329_list, data.table, stringsAsFactors=FALSE))
COVID19_0329_unique<-unique(COVID19_0329)



# 04-01, 04-03, 04-05 (2020-04-05 23:06:45)
get_recent_report<- function(dates){
  
  listofdfs <- list() #Create a list in which you intend to save your df's.
  
  for (i in 1:length(dates)){ #Loop through the numbers of dates instead of the dates
    
    #Use dates[i] instead of i to get the date
    
    daily_url<-paste0("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/", dates[i], ".csv")
    df <-
      data.table(read_csv(url(daily_url)))
    keep_variables<- c("Province_State", "Country_Region", "Last_Update", "Confirmed", "Deaths", "Recovered")
    COVID19_by_date<- df[,..keep_variables]
    COVID19_by_date[,Date:=as.Date(Last_Update, "%m/%d/%Y")] 
    listofdfs[[i]] <- COVID19_by_date[,.(Confirmed, Deaths, Recovered), by=.(Province_State, Country_Region, Date)] # save your dataframes into the list
  }
  
  return(listofdfs) #Return the list of dataframes.
}

dateids<- c("04-01-2020", "04-03-2020", "04-05-2020") 
COVID19_0401_list<- get_recent_report(dates = dateids)
# unlist as data.table
COVID19_0401<-do.call(rbind, lapply(COVID19_0401_list, data.table, stringsAsFactors=FALSE))
COVID19_0401_unique<-unique(COVID19_0401)



# 04-06
daily_url<-paste0("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/04-06-2020.csv")
df <-
  data.table(read_csv(url(daily_url)))
keep_variables<- c("Province_State", "Country_Region", "Last_Update", "Confirmed", "Deaths", "Recovered")
COVID19_0406<- df[,..keep_variables]
COVID19_0406<-COVID19_0406[,Date:=as.Date(Last_Update, "%m/%d/%y")][,.(Confirmed, Deaths, Recovered), by=.(Province_State, Country_Region, Date)]
COVID19_0406_unique<-unique(COVID19_0406)

# 04-07~now
get_recent_report<- function(dates){
  
  listofdfs <- list() #Create a list in which you intend to save your df's.
  
  for (i in 1:length(dates)){ #Loop through the numbers of dates instead of the dates
    
    #Use dates[i] instead of i to get the date
    
    daily_url<-paste0("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/", dates[i], ".csv")
    df <-
      data.table(read_csv(url(daily_url)))
    keep_variables<- c("Province_State", "Country_Region", "Last_Update", "Confirmed", "Deaths", "Recovered")
    COVID19_by_date<- df[,..keep_variables]
    COVID19_by_date[,Date:=as.Date(Last_Update, "%m/%d/%Y")] 
    listofdfs[[i]] <- COVID19_by_date[,.(Confirmed, Deaths, Recovered), by=.(Province_State, Country_Region, Date)] # save your dataframes into the list
  }
  
  return(listofdfs) #Return the list of dataframes.
}

dateids<- as.character(format(as.Date(seq(as.Date("2020-04-07"), Sys.Date()-1, by=1)), "%m-%d-%Y"))
COVID19_recent_list<- get_recent_report(dates = dateids)
# unlist as data.table
COVID19_recent<-do.call(rbind, lapply(COVID19_recent_list, data.table, stringsAsFactors=FALSE))
COVID19_recent_unique<-unique(COVID19_recent)

# rbind together
COVID19_daily_list<- rbindlist(list(COVID19_0122_unique, COVID19_early_unique, COVID19_0201_unique, COVID19_0323_unique, COVID19_0322_unique, 
                                    COVID19_mid_unique, COVID19_0329_unique, COVID19_recent_unique,COVID19_0401_unique,COVID19_0406_unique))

# US
COVID19_daily_US<- 
  COVID19_daily_list[,.(Province_State, Country_Region, Date, Confirm_count=Confirmed, Death_count=Deaths, Recover_count=Recovered)][order(Date)&Country_Region=="US"]

# Get rid of city 
us_CommaState<- dput(grep(',',unique(COVID19_daily_US[Country_Region == "US",Province_State]),value = T, ignore.case=T))
us_state<- setdiff(COVID19_daily_US[Country_Region == "US",Province_State],us_CommaState)
COVID19_us<-
  COVID19_daily_US[Province_State %in% us_state]


COVID19_us_state<-
  unique(COVID19_us[,`:=`(sum_confirm=sum(Confirm_count),
                          sum_death=sum(Death_count),
                          sum_recover=sum(Recover_count)),
                    by=.(Province_State, Country_Region, Date)][,.(Province_State, Country_Region, Date, sum_confirm, sum_death)])[order(Date)]

# Create flag for cruise
COVID19_us_state[,cruise:=0]
COVID19_us_state[Province_State%in% 
                   dput(grep("ship|Princess", COVID19_us_state$Province_State, value=TRUE)),
                 cruise:=1]

COVID19_us_state<-COVID19_us_state[cruise==0][,.(Province_State, Country_Region, Date, Total_confirm=sum_confirm, Total_death=sum_death)]


# merge with COVID19_us_state
COVID19_US<-merge(COVID19_us_state,US_pop, by.x="Province_State", by.y="NAME", all.x=T)

## output RData object
write.csv(COVID19_US, file=paste0('/home/ec2-user/covid_data/COVID19_USJH','.csv'))

