#### This is an add-hoc model which I created as an example of recommender systems.

### Dataset
ml-100k movie ranks dataset.

You can download the dataset from https://grouplens.org/datasets/movielens/100k/

### Model Description

In fact, this doesn't use any statistical calculations (i.e ALS, Pearson Correleation, Cosine Similarities) to find "movies to be liked" but it is effective as the train dataset grows. 

Model takes object user an input from the terminal and finds other users who rated 4-5 points to the same film to which object user rated 5 points. Then it finds those users' other film rankings and calculates top n movies higly ranked ones.

Model uses Spark RDD data structure and it is a good practice to work on Spark RDDs due to having many different data editing operations.

### Simple Usage


![](Utils/Screenshot%20from%202019-02-23%2014-51-49.png)

Note that "100" represents the object user to whom we want to find good movies.

When it is run, model prints some checkpoints and finally top 10 results with their density scores which is the count of times that the film was rated 4 or 5 points from other users.
![](Utils/Screenshot%20from%202019-02-23%2015-00-43.png)

