#!/usr/bin/env python
# -*- coding: utf-8 -*-

import getpass
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RankingEvaluator
import pyspark.sql.functions as func 
from pyspark.ml.recommendation import ALS
import time

def model_ALS(train, rank, reg, a):
    als = ALS(rank=rank, maxIter=10, regParam=reg, alpha=a, userCol="user_id", itemCol="track_id", ratingCol="count", coldStartStrategy="drop", implicitPrefs=True)
    model = als.fit(train)
    return model 

if __name__ == "__main__":
    
    spark = SparkSession.builder.appName('final-project').getOrCreate()
    username = getpass.getuser()

    train = spark.read.parquet(f'/user/{username}/1004-project-2023/train_inter_small_3.parquet', schema='user_id INT, track_id DOUBLE, count INT')
    train.createOrReplaceTempView('train')

    valid = spark.read.parquet(f'/user/{username}/1004-project-2023/vali_inter_small_3.parquet', schema='user_id INT, true ARRAY<DOUBLE>').sort('user_id') \
                 .withColumn('true_double', func.col('true').cast("array<double>"))\
                 .drop('true')
    test = spark.read.parquet(f'/user/{username}/1004-project-2023/test_inter_3.parquet', schema='user_id INT, track_id DOUBLE, count INT')
    #test.show()
    
    ranks = [10, 50, 100]
    regs = [0.01, 0.1, 0.5]
    alphas = [0.25, 0.5, 0.75]
    
    best_map=0
    best_rank=0
    best_reg=0
    best_alpha=0
    
    unique_users = valid.select("user_id").distinct()

    for rank in ranks:
      for reg in regs:
        for alpha in alphas:

            start = time.time()
            als= model_ALS(train, rank, reg, alpha)  
            end = time.time()
            print("time to train", end - start)
            
            start = time.time()
            vali_rec = als.recommendForUserSubset(unique_users, 100)
            end = time.time()
            print("time to recommend", end - start)
            
            # time to evaluate
            start = time.time()
            pred = vali_rec.select(vali_rec['user_id'], vali_rec.recommendations['track_id'].alias('pred')) \
                           .withColumn('pred_double', func.col('pred').cast("array<double>"))\
                           .drop('pred')
            pred_true = pred.join(valid, 'user_id')
            #pred_true.show()
            
            evaluator = RankingEvaluator(predictionCol='pred_double', labelCol='true_double')
            val_map = evaluator.evaluate(pred_true, {evaluator.metricName: 'meanAveragePrecisionAtK', evaluator.k: 100})
            end = time.time()
            print("time to find map", end - start)
            
            print(rank, reg, alpha, val_map)
            cur_res = spark.createDataFrame([(rank, reg, alpha, val_map)], ['rank', 'reg', 'alpha', 'map'])
            cur_res.write.mode('append').parquet(f'/user/{username}/1004-project-2023/train_val_res.parquet')
            cur_res = spark.read.parquet(f'/user/{username}/1004-project-2023/train_val_res.parquet')
            cur_res.show()
            
            if val_map > best_map:
                best_map = val_map
                best_rank = rank
                best_reg = reg
                best_alpha = alpha

    #test using best model
    als= model_ALS(train, best_rank, best_reg, best_alpha)
    test_rec = als.recommendForUserSubset(test.select("user_id").distinct(), 100)
    pred = test_rec.select(test_rec['user_id'], test_rec.recommendations['track_id'].alias('pred'))
    pred_true = pred.join(test, 'user_id').rdd.map(lambda row: (row[1], row[2])) 
    #spark.createDataFrame(true_pred).show()
    
    test_map = RankingMetrics(true_pred).meanAveragePrecisionAt(100)
    print(best_rank, best_reg, best_alpha, al_map)
    cur_res = spark.createDataFrame([(best_rank, best_reg, best_alpha, al_map)], ['rank', 'reg', 'alpha', 'map'])
    cur_res.write.mode('append').parquet(f'/user/{username}/1004-project-2023/train_val_res.parquet')
    cur_res = spark.read.parquet(f'/user/{username}/1004-project-2023/train_val_res.parquet')
    cur_res.show()
    
    spark.stop()
