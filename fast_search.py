import getpass
from pyspark.sql import SparkSession
from pyspark.mllib.evaluation import RankingMetrics
import pyspark.sql.functions as func
from pyspark.ml.recommendation import ALS, ALSModel
import time
from annoy import AnnoyIndex

def model_ALS(train, rank, reg, a):
    als = ALS(rank=rank, maxIter=10, regParam=reg, alpha=a, userCol="user_id", itemCol="track_id", ratingCol="count", coldStartStrategy="drop", implicitPrefs=True)
    model = als.fit(train)
    return model

def create_annoy_index(model, num_trees):
    item_factors = model.itemFactors
    annoy_index = AnnoyIndex(len(item_factors.select("features").first()["features"]), 'angular')
    for item in item_factors.collect():
        annoy_index.add_item(item[0], item[1])
    annoy_index.build(num_trees)
    return annoy_index

def get_annoy_recommendations(model, annoy_index, num_items):
    rec_by_user = []
    for userFactor in model.userFactors.collect():
        user_vector = userFactor[1]
        rec = annoy_index.get_nns_by_vector(user_vector, num_items)
        rec_by_user.append((userFactor[0], rec))
    return rec_by_user

if __name__ == "__main__":

    spark = SparkSession.builder.appName('final-project').getOrCreate()
    username = getpass.getuser()

    train = spark.read.parquet(f'/user/{username}/1004-project-2023/train_inter_FS.parquet', schema='user_id INT, track_id INT, count INT')
    train.createOrReplaceTempView('train')

    valid = spark.read.parquet(f'/user/{username}/1004-project-2023/vali_inter_FS.parquet')
    valid.createOrReplaceTempView('valid')
    unique_users = valid.select("user_id").distinct()

    ### ALS
    rank = 10
    reg = 0.1
    alpha = 0.5

    start = time.time()
    als= model_ALS(train, rank, reg, alpha)  
    end = time.time()
    print("time to train", end - start)
    
    start = time.time()
    vali_rec = als.recommendForUserSubset(unique_users, 100)
    end = time.time()
    print("time to recommend", end - start)
    
    start = time.time()
    pred = vali_rec.select(vali_rec['user_id'], vali_rec.recommendations['track_id'].alias('pred'))
    pred_true_als = pred.join(valid, 'user_id').rdd.map(lambda row: (row[1], row[2]))
    val_map_als = RankingMetrics(pred_true_als).meanAveragePrecisionAt(100)
    end = time.time()
    print("time to find map", end - start)

    print("ALS MAP:", val_map_als)
    
    ### Annoy
    num_trees = 50
    num_items = 10

    start = time.time()
    annoy_index = create_annoy_index(als, num_trees)
    annoy_index.save('annoy_index.ann')
    end = time.time()
    print("time to create Annoy index", end - start)

    start = time.time()
    annoy_rec = get_annoy_recommendations(als, annoy_index, num_items)
    end = time.time()
    print("time to recommend using annoy", end - start)
    
    # To evaluate
    start = time.time()
    pred_annoy = spark.createDataFrame(annoy_rec, ["user_id", "pred"])
    pred_true_annoy = pred_annoy.join(valid, "user_id").rdd.map(lambda row: (row[1], row[2]))
    val_map_annoy = RankingMetrics(pred_true_annoy).meanAveragePrecisionAt(10)
    end = time.time()
    print("time to find map", end - start)

    print("Annoy MAP:", val_map_annoy)
    
    spark.stop()

