#!/usr/bin/env python
# -*- coding: utf-8 -*-

import getpass
from pyspark.sql import SparkSession
from pyspark.sql.functions import collect_list
from pyspark.mllib.evaluation import RankingMetrics

def preprocessing(spark, username, file):
    
    data = spark.read.parquet(f'/user/{username}/1004-project-2023/{file}.parquet', schema='user_id INT, track_id INT, count INT')
    data.createOrReplaceTempView('data')
    return data
    
def model_popularity(data, beta):
    
    pred = spark.sql(f'SELECT track_id AS pred_top FROM data GROUP BY track_id ORDER BY COUNT(DISTINCT user_id)/(SUM(count) + {beta}) DESC LIMIT 100') \
           .agg(collect_list('pred_top').alias('pred_top'))
    
    return pred

def true_popularity(data):
    
    true = spark.sql('SELECT * FROM data ORDER BY user_id, count DESC') \
           .groupBy('user_id') \
           .agg(collect_list('track_id').alias('true_top')).select('true_top')
    
    return true

if __name__ == "__main__":
    
    spark = SparkSession.builder.appName('final-project').getOrCreate()
    username = getpass.getuser()
    
    train_inter = preprocessing(spark, username, 'train_inter_small_2')
    vali_inter = preprocessing(spark, username, 'vali_inter_small_2')
    test_inter = preprocessing(spark, username, 'test_inter_2')
    
    train_true = true_popularity(train_inter)
    vali_true = true_popularity(vali_inter)
    test_true = true_popularity(test_inter)
    
    betas = [0, 1000, 10000, 100000, 200000, 500000, 1000000]
    vali_results = {}
    
    for beta in betas:
        pred = model_popularity(train_inter, beta)
	
        train_true_pred = train_true.crossJoin(pred)
        vali_true_pred = vali_true.crossJoin(pred)
        
        train_ranking_metrics = RankingMetrics(train_true_pred.rdd)
        vali_ranking_metrics = RankingMetrics(vali_true_pred.rdd)
        
        train_map = train_ranking_metrics.meanAveragePrecisionAt(100)
        vali_map = vali_ranking_metrics.meanAveragePrecisionAt(100)
        
        print(f'popularity baseline MAP at 100 on train set = {train_map} with beta = {beta}')
        print(f'popularity baseline MAP at 100 on validation set = {vali_map} with beta = {beta}')
    
        vali_results[beta] = vali_map
    
    best_beta = max(vali_results, key=vali_results.get)
    best_map = vali_results[best_beta]
    print(f'Best beta: {best_beta}, validation MAP at 100: {best_map}')
    
    pred = model_popularity(train_inter, best_beta)
    test_true_pred = test_true.crossJoin(pred)
    test_ranking_metrics = RankingMetrics(test_true_pred.rdd)
    test_map = test_ranking_metrics.meanAveragePrecisionAt(100)
    print(f'popularity baseline MAP at 100 on test set = {test_map} with beta = {best_beta}')
    
    spark.stop()
