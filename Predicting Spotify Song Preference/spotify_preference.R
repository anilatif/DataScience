#aneeka latif
#ist719
#data report
#load in the spotify data
spot_data <- read.csv("downloads/spot_data_shortened.csv",
                      sep=",", header=TRUE,
                      stringsAsFactors = FALSE)

extra_anno <- read.csv("downloads/download.csv",
                       sep=",", header=TRUE,
                       stringsAsFactors = FALSE)


summary(spot_data)

#get the extra annotations from amazon mechanical turk
anno2 <- extra_anno$Answer.category.label

#clean the genre column, make pop == Pop 
spot_data[spot_data$Genre=="pop","Genre"] = "Pop"


#remove the "" quotes
anno2 <- gsub("\"", "", anno2)

#make a new dataframe of genre annotations with my labels + AMT labels
anno_df <- data.frame(spot_data$Genre, anno2)

#install.packages("psych")
library("psych")
cohen.kappa(anno_df, w=NULL,n.obs=length(anno_df),alpha=.05,levels=NULL) 

#get the percentage breakdown of likeable vs unlikable songs
lik_song = round(sum(spot_data$target==1)/
                   length(spot_data$target),2)*100
dont_lik_song = round(sum(spot_data$target==0)/
                        length(spot_data$target),2)*100

#make lables from percentage + descriptor
lik_song_label =paste("Songs Liked",lik_song,"%")
dont_lik_song_label =paste("Songs Disliked", dont_lik_song,"%")


#make a pie chart of likeability as graph #1 
pie(table(spot_data$target), 
    labels=c(dont_lik_song_label, lik_song_label),
    col=c("#1ed760", "#ffffff"))


#____________________________________________________________

#make a density graph of tempo as graph #2 
d <- density(spot_data$tempo)
plot(d, main="Distribution of Tempo", xlab="Tempo", color="white")
polygon(d, col="#dddddd")

library(ggplot2)
##redo the graph as a ggplot

ggplot(spot_data, aes(x=spot_data$tempo))+
  geom_density(fill="white", color="white")+labs(x="Tempo")


#____________________________________________________________
library (RColorBrewer)
#make a bar chart of key as graph #3
hist(spot_data$key, main="Distribution of Key in a 
     Sample of Spotify songs", xlab="Key",
     col=c("#1db954","#00FF00","#ffffff", "#191414"))

#____________________________________________________________
#make multidimensional plots for graphs 3 and 

par(mfrow=(c(2,1)))

plot(spot_data$energy,spot_data$loudness, main="Relative Energy as a Function of Loudness",
     xlab="Energy (scaled)", ylab="Loudness (Decibels)", col="#1db954")

plot(spot_data$danceability,spot_data$tempo, main="Relative Danceability as a Function of Tempo",
     xlab="Energy (scaled)", ylab="Tempo (BPM))")





library(ggplot2)
ggplot(spot_data,aes(x=spot_data$tempo))+geom_histogram(bins=10)+facet_grid(~target)+theme_bw()

ggplot(spot_data,aes(x=spot_data$acousticness))+geom_histogram(bins=10)+facet_grid(~target)+theme_bw()




##################################################
# Question 1 #is there an artist bias?

library(wordcloud)
#stopwords method is not working
#stopwords = set(c("the"))

# build a list to feed the wordcloud, remove "the" and other unnecessary words
liked_artist <- spot_data[spot_data$target == 1, "artist"]
#liked_artist <- gsub("The", "", liked_artist)

#i want to keep the artists names together 
#so i used a character that doesnt encode properly
#it leaves little boxes but I may erase or fill with a symbol in illustrator
liked_artist <- gsub(" ", "。", liked_artist)

wordcloud(liked_artist, 
          colors=c("#dddddd","#006400", 
                   "#00FF00", "#191414"))



## do the same thing for disliked artists 
# build a list to feed the wordcloud, remove "the" and other unnecessary words
disliked_artist <- spot_data[spot_data$target == 0, "artist"]
disliked_artist <- gsub(" ", "。", disliked_artist)
wordcloud(disliked_artist, 
          colors=c("#363636","#949494", "#cccccc" ))


                
                          

##############################################################

#is there a genre bias? 
liked_genre <- spot_data[spot_data$target == 1, "Genre"]
liked_genre_df <- data.frame(liked_genre, 1)

ggplot() + geom_bar(data=spot_data, 
                    aes(x==spot_data[spot_data$target == 1, 
                                     "Genre"]))

#set up a new column for categorical liked/disliked data
spot_data$like <- "liked"
spot_data[spot_data$target == 0,"like"] <- "disliked"

ggplot(data=spot_data, aes(x=Genre, fill=like)) +
  geom_bar(stat="count", color="white", position=position_dodge())+
  theme_minimal() + scale_fill_manual(values=c("#cccccc","#006400"))+
  labs(ylab="Count")
#"#006400"

#what is the makeup of liked/disliked songs?
library(ggplot2)

ggplot(spot_data, aes(x=instrumentalness, y=loudness, fill=like, color=danceability, size=tempo^2)) +
  geom_point(shape=21, stroke=2, alpha=.7) +
  scale_color_gradient(low="red", high="green") +
  scale_size_continuous(range=c(1,12))


boxplot(spot_data[spot_data$target == 1, "danceability"])
boxplot(spot_data[spot_data$target == 0, "danceability"])

par(mfrow = c(2,2))
boxplot(danceability~like, data=spot_data, border=c("#949494", "#006400"),
        main = "Danceability")
boxplot(instrumentalness~like, data=spot_data, border=c("#949494", "#006400"),
        main = "Instrumentalness")
boxplot(loudness~like, data=spot_data, border=c("#949494", "#006400"),
        main = "Loudness")
boxplot(tempo~like, data=spot_data, border=c("#949494", "#006400"),
        main="Tempo")




#can we predict liked songs? 
library(caret)
songModel <- train(Genre ~ ., data=spot_data, method = "knn")
#studentTestPred <- predict(studentModel, studentTest) 
#confusionMatrix(studentTestPred, studentTest$performance)$overall['Accuracy'] 
#Accuracy -  0.5897

#

library(rpart)
library(rpart.plot)
my_tree <- rpart(like~ danceability + energy + 
                   key+loudness + speechiness + acousticness + instrumentalness+ 
                   liveness+ valence+ Genre+ tempo, data = spot_data,
                 method = "class")

# Visualize the decision tree using plot() and text()
#use rpart.plot
rpart.plot(my_tree, box.palette="GyGn")



