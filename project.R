download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv","./project/train.csv")
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv","./project/test.csv")
train<-read.csv("./project/train.csv")
test<-read.csv("./project/test.csv")

names(train)
head(train$X)
