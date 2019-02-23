import sys
from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import Rating

conf = SparkConf().setMaster("local[*]").setAppName("MovieRecommendationsALS")
sc = SparkContext(conf = conf)
sc.setCheckpointDir('checkpoint')
sc.setLogLevel("ERROR")

def loadMovieNames():
    movieNames = {}
    with open("/Users/ozgurcengelli/Desktop/ml-100k/u.ITEM", encoding = "latin-1") as f:
        for line in f:
            fields = line.split('|')
            movieNames[int(fields[0])] = fields[1]
    return movieNames
print("\n Loading movie archive")
nameDict = loadMovieNames()

data = sc.textFile("file:/Users/ozgurcengelli/Desktop/ml-100k/u.data")

ratings = data.map(lambda l: l.split()).map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2]))).cache()
userID = int(sys.argv[1])

#here, we picked the films to which the selected user gave 5-point.
print("\n Analysing the ratings of the selected user")
userRatings = ratings.filter(lambda l: l[0] == userID and l[2] == 5.0)
films = list(userRatings.map(lambda l: l[1]).collect())

def filmbul(array):
    if array[1] in films:
        return array
    else:
        return "NaN"

print("\n Finding similar users")
#here, other users are chosen who gave similar ranks (4 or 5) for the movies to which the selected user gave 5-point.
otherRatings = ratings.filter(lambda x: filmbul(x) != "NaN" and (x[2]==5 or x[2]==4))
similarUsers = otherRatings.map(lambda x: (x[0],1)).groupByKey().mapValues(len)
users = list(similarUsers.map(lambda l: l[0]).collect())

#here, we picked films which rated with 4 or 5 point by the other users and weren't rated by the selected user.

def findSimilarUser(array):
    if array[0] in users:
        return array
    else:
        return "NaN"

usersfilms = list(ratings.filter(lambda l: l[0] == userID).map(lambda x: x[1]).collect())

def findusersfilms(array):
    if array[1] in usersfilms:
        return array
    else:
        return "NaN"

print("\n Analysing similar users' ratings")
bestFilms = ratings.filter(lambda x: findSimilarUser(x) != "NaN" and (x[2]==5 or x[2] == 4) and findusersfilms(x) == "NaN").map(lambda x: (x[1],1))

#finally, we select the films which have top 10 highest density score. 
print("\n Top 10 recommendations:")
top10BestFilms = bestFilms.groupByKey().mapValues(len).sortBy((lambda x: x[1]), ascending=False).take(10)
for i in top10BestFilms:
    print(nameDict[int(i[0])], "with the density score of", i[1])
