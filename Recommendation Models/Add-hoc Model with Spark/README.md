#### This is an add-hoc model which I created as an example of recommender systems.

### Dataset
ml-100k movie ranks dataset.

You can download the dataset from https://grouplens.org/datasets/movielens/100k/

### Model Description

Model, in fact, doesn't use any statistical calculations (i.e ALS, Pearson Correleation, Cosine Similarities) to find "movies to be liked" but it is effective as the train dataset grows. 

It simply finds other users that ranked films similarly to object user. Then it finds those users' ranks that gave to other movies and calculates top n movies higly ranked ones.
