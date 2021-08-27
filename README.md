# covid19_model
This repository serves to house work done between 7/2019 and 12/2019 toward the development and distribution of a covid-19 model within the VHA. Several references to VA data resources have been removed for privacy concerns, as well as the dynamic output of the model as a power-BI dashboard for the same reason. The project itself contained several steps, beginning with gathering data published by JHU and merging this with VA data. This was then used to train and deploy several models, with the arima + wavelet model being the most successful, and a highly tuned XGB model being close behind, but notably slower. Both involved careful assessments of prediction accuracy around large events, responsiveness to underlying data generating process changes (via synthetic generation and testing _link_) and substantial attention paid to correlations between neighboring geographical regions. Temporo-spatial methods are not shown due to PHI concerns. However, once completed, the model was used by many VA facilities for local operational planning. 

While the model has since been retired, the code will be retained here for future reference and possible reactivation as needed.
## Sample output 1:
![](splash_images/sample_bi_dashboard.png?raw=true)
## Sample output 2:
![](splash_images/sample_bi_dashboard_2.png?raw=true)
