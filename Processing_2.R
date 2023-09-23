dt=read.csv("train.csv")
dt.test=read.csv("test_X.csv")
names(dt)
########################################train######################################
###################################################################################

hist(dt$Age)
par(mfrow=c(2,1))
hist(dt$RestingBP[dt$HeartDisease==0],xlim = c(80,180),breaks = 20,ylim = c(0,70))
hist(dt$RestingBP[dt$HeartDisease==1],xlim = c(80,180),breaks = 20,ylim = c(0,70))

hist(dt$Cholesterol)#0 values in ~100 observations
hist(dt$MaxHR)
hist(dt$Oldpeak)

table(dt$FastingBS)
table(dt$Sex)
table(dt$ChestPainType)
table(dt$RestingECG)
table(dt$ExerciseAngina)
table(dt$ST_Slope)


dt.com=dt[dt$Cholesterol!=0,]
m1=glm(HeartDisease~Age+factor(Sex)+factor(ChestPainType)+
         RestingBP+Cholesterol+factor(FastingBS)+factor(RestingECG)+
         MaxHR+factor(ExerciseAngina)+Oldpeak+factor(ST_Slope),family = "binomial",data=dt.com)
summary(m1)
#age, sex,chestpain type, exerciseangina, oldpeak, stslope


dt.na=dt[dt$Cholesterol==0,]
m1=glm(HeartDisease~Age+factor(Sex)+factor(ChestPainType)+
         RestingBP+factor(FastingBS)+factor(RestingECG)+
         MaxHR+factor(ExerciseAngina)+Oldpeak+factor(ST_Slope),family = "binomial",data=dt.na)
summary(m1)
#chest pain, fasting BS, MAxHR,exeriseAngina


m1=glm(HeartDisease~Age+factor(Sex)+factor(ChestPainType)+
         RestingBP+factor(FastingBS)+factor(RestingECG)+
         MaxHR+factor(ExerciseAngina)+Oldpeak+factor(ST_Slope),family = "binomial",data=dt)
summary(m1)
#sex, chestpain(ATA+NAP),fasting BS,  exerciseangina,oldpeak,st_slope

####one-hot encoding
Sex<-data.frame(model.matrix(~Sex-1, data=dt))
ChestPainType<-data.frame(model.matrix(~ChestPainType-1, data=dt))
RestingECG<-data.frame(model.matrix(~RestingECG-1, data=dt))
ExerciseAngina<-data.frame(model.matrix(~ExerciseAngina-1, data=dt))
ST_Slope<-data.frame(model.matrix(~ST_Slope-1, data=dt))
dt$FastingBS_c<-ifelse(dt$FastingBS==1,"high","low")
dt$FastingBS_c<-as.factor(dt$FastingBS_c)
FastingBS<-data.frame(model.matrix(~FastingBS_c-1, data=dt))



dt$Age_c<-ifelse(dt$Age>=65,"old","young")
Age<-data.frame(model.matrix(~Age_c-1, data=dt))

dt$RestingBP_c<-ifelse(dt$RestingBP>=140,"high","low")
RestingBP<-data.frame(model.matrix(~RestingBP_c-1, data=dt))


par(mfrow=c(2,1))
hist(dt$Oldpeak[dt$HeartDisease==0],xlim = c(-2,6),ylim=c(0,100),breaks = 20)
hist(dt$Oldpeak[dt$HeartDisease==1],xlim = c(-2,6),ylim=c(0,100),breaks = 40)
#,xlim = c(80,180),ylim=c(0,100)
dt$Oldpeak_c<-ifelse(dt$Oldpeak>0.5,"high","low")
Oldpeak_c<-data.frame(model.matrix(~Oldpeak_c-1, data=dt))



selected<-dt[, c("PatientID","Oldpeak")]
HeartDisease<-dt[,c("HeartDisease")]

dt.use=cbind(selected,Oldpeak_c,Sex, ChestPainType, FastingBS,RestingECG,ExerciseAngina,ST_Slope,HeartDisease)

write.csv(dt.use, file = "train_processed_dropped_features_3.csv")
#write.csv(dt.use[dt.use$Cholesterol!=0,],file = "train_processed_com.csv")




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



selected<-dt.test[, c("PatientID","Oldpeak")]

dt.use=cbind(selected,RestingBP,Sex, ChestPainType, FastingBS,RestingECG,ExerciseAngina,ST_Slope)


dt.test$RestingBP_c<-ifelse(dt.test$RestingBP>=140,"high","low")
RestingBP<-data.frame(model.matrix(~RestingBP_c-1, data=dt.test))


write.csv(dt.use,file = "test_processed_dropped_features_1.csv")



###########