dt=read.csv("train.csv")
dt.test=read.csv("test_X.csv")
names(dt)
########################################train######################################
###################################################################################

hist(dt$Age)
hist(dt$RestingBP)
hist(dt$Cholesterol)#0 values in ~100 observations
hist(dt$MaxHR)
hist(dt$Oldpeak)

table(dt$FastingBS)
table(dt$Sex)
table(dt$ChestPainType)
table(dt$RestingECG)
table(dt$ExerciseAngina)
table(dt$ST_Slope)

m1=glm(HeartDisease~Age+factor(Sex)+factor(ChestPainType)+
         RestingBP+Cholesterol+factor(FastingBS)+factor(RestingECG)+
         MaxHR+factor(ExerciseAngina)+Oldpeak+factor(ST_Slope),family = "binomial",data=dt)
summary(m1)

####one-hot encoding
Sex<-data.frame(model.matrix(~Sex-1, data=dt))
ChestPainType<-data.frame(model.matrix(~ChestPainType-1, data=dt))
RestingECG<-data.frame(model.matrix(~RestingECG-1, data=dt))
ExerciseAngina<-data.frame(model.matrix(~ExerciseAngina-1, data=dt))
ST_Slope<-data.frame(model.matrix(~ST_Slope-1, data=dt))
dt$FastingBS_c<-ifelse(dt$FastingBS==1,"high","low")
dt$FastingBS_c<-as.factor(dt$FastingBS_c)
FastingBS<-data.frame(model.matrix(~FastingBS_c-1, data=dt))


####scale the data
#clean all 0 values in chol and BP to NA
dt$Cholesterol_clean=dt$Cholesterol
dt$Cholesterol_clean[dt$Cholesterol_clean==0]<-NA
dt$Cholesterol_s=scale(dt$Cholesterol_clean)
dt$Cholesterol_clean[dt$Cholesterol_s>=2.8]=max(dt$Cholesterol_clean[dt$Cholesterol_s<2.8],na.rm = TRUE)
dt$Cholesterol_s=scale(dt$Cholesterol_clean)
#mean(dt$Cholesterol_clean,na.rm = TRUE) 242.6368
#sd(dt$Cholesterol_clean,na.rm = TRUE) 54.38769

dt$RestingBP_clean=dt$RestingBP
dt$RestingBP_clean[dt$RestingBP_clean==0]<-NA
dt$RestingBP_s=scale(dt$RestingBP_clean)
#dt$RestingBP_clean[dt$RestingBP_s>=2.8]=max(dt$RestingBP_clean[dt$RestingBP_s<2.8],na.rm = TRUE)
#dt$RestingBP_clean[dt$RestingBP_s<=-2.8]=min(dt$RestingBP_clean[dt$RestingBP_s>-2.8],na.rm = TRUE)
#dt$RestingBP_s=scale(dt$RestingBP_clean)
#mean(dt$RestingBP_clean) 132.8042
#sd(dt$RestingBP_clean) 18.47636

#age, HR, Oldpeak
dt$Age_s=scale(dt$Age)
dt$MaxHR_s=scale(dt$MaxHR)
dt$Oldpeak_s=scale(dt$MaxHR)


selected<-dt[, c("PatientID","Age_s","RestingBP_s","Cholesterol_s","MaxHR_s","Oldpeak_s")]
HeartDisease<-dt[,c("HeartDisease")]

dt.use=cbind(selected,Sex, ChestPainType, FastingBS,RestingECG,ExerciseAngina,ST_Slope,HeartDisease)
write.csv(dt.use,file = "train_processed.csv")




#
################################################test########################################
############################################################################################
#for test data
####one-hot encoding
Sex<-data.frame(model.matrix(~Sex-1, data=dt.test))
ChestPainType<-data.frame(model.matrix(~ChestPainType-1, data=dt.test))
RestingECG<-data.frame(model.matrix(~RestingECG-1, data=dt.test))
ExerciseAngina<-data.frame(model.matrix(~ExerciseAngina-1, data=dt.test))
ST_Slope<-data.frame(model.matrix(~ST_Slope-1, data=dt.test))
dt.test$FastingBS_c<-ifelse(dt.test$FastingBS==1,"high","low")
dt.test$FastingBS_c<-as.factor(dt.test$FastingBS_c)
FastingBS<-data.frame(model.matrix(~FastingBS_c-1, data=dt.test))

####scale the data
#clean all 0 values in chol and BP to NA
dt.test$Cholesterol_clean=dt.test$Cholesterol
dt.test$Cholesterol_clean[dt.test$Cholesterol_clean==0]<-NA
dt.test$Cholesterol_s=(dt.test$Cholesterol_clean-242.6368)/54.38769
#dt.test$Cholesterol_clean[dt.test$Cholesterol_s>=2.8]=max(dt.test$Cholesterol_clean[dt.test$Cholesterol_s<2.8],na.rm = TRUE)


dt.test$RestingBP_clean=dt.test$RestingBP
dt.test$RestingBP_clean[dt.test$RestingBP_clean==0]<-NA
dt.test$RestingBP_s=(dt.test$RestingBP_clean-132.8042)/18.47636
#dt.test$RestingBP_clean[dt.test$RestingBP_s>=2.8]=max(dt.test$RestingBP_clean[dt.test$RestingBP_s<2.8],na.rm = TRUE)
#dt$RestingBP_clean[dt$RestingBP_s<=-2.8]=min(dt$RestingBP_clean[dt$RestingBP_s>-2.8],na.rm = TRUE)
#dt$RestingBP_s=scale(dt$RestingBP_clean)


#age, HR, Oldpeak
dt.test$Age_s=(dt.test$Age-53.92083)/9.366327
dt.test$MaxHR_s=(dt.test$MaxHR-136.0917)/25.16753
dt.test$Oldpeak_s=(dt.test$MaxHR-0.92)/1.081114


selected<-dt.test[, c("PatientID","Age_s","RestingBP_s","Cholesterol_s","MaxHR_s","Oldpeak_s")]
dt.use=cbind(selected,Sex, ChestPainType, FastingBS,RestingECG,ExerciseAngina,ST_Slope)
write.csv(dt.use,file = "test_processed.csv")



###########