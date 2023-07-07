#!/usr/bin/env python
# -*- coding: utf-8 -*-

import getpass
from pyspark.sql import SparkSession
from pyspark.sql.functions import row_number, col, monotonically_increasing_id, max, min, dense_rank, count, collect_set
from pyspark.sql.window import Window

def trainvalisplit(spark, username):
    interactions_train = spark.read.parquet(f'/user/{username}/1004-project-2023/interactions_train_small.parquet', schema='user_id INT, recording_msid STRING, timestamp TIMESTAMP')
    interactions_train.createOrReplaceTempView('interactions_train')
    interactions_train = interactions_train.drop('timestamp')
    interactions_test = spark.read.parquet(f'/user/{username}/1004-project-2023/interactions_test.parquet', schema='user_id INT, recording_msid STRING, timestamp TIMESTAMP')
    interactions_test.createOrReplaceTempView('interactions_test')
    interactions_test = interactions_test.drop('timestamp')
    
    # filter user less than 10 interactions
    #print("rows in train before filter:", interactions_train.count())
    interactions_count = interactions_train.groupBy('user_id').agg(count('*').alias('num_interactions'))
    selected_user_ids = interactions_count.filter(interactions_count.num_interactions >= 50).select('user_id').rdd.flatMap(lambda x: x).collect()
    train_filtered = interactions_train.filter(interactions_train.user_id.isin(selected_user_ids))
    #print("rows in train after filter:", train_filtered.count())
    
    # given track_id to recording_msid
    unique_msid = train_filtered.select("recording_msid").union(interactions_test.select("recording_msid")).distinct()
    msid_trackid = unique_msid.rdd.zipWithIndex().map(lambda x: (x[0][0], x[1]))
    df_trackid = msid_trackid.toDF(['recording_msid', 'track_id'])
    #df_trackid.show()
    
    # random split into train and validation, due to filtering the probability of cold start is closer to 0
    train_with_id = train_filtered.withColumn("interaction_id", monotonically_increasing_id()) 
    sample_fraction = {category: 0.8 for category in selected_user_ids}
    train = train_with_id.sampleBy("user_id", sample_fraction, seed=42)
    #print("rows in train:", train.count())
    valid = train_with_id.subtract(train)
    #print("rows in valid:", valid.count())
    
    # create R matrix
    group_cols = ['user_id', 'recording_msid']
    train_R_msid = train.drop('interaction_id').groupBy(group_cols).count()
    valid_R_msid = valid.drop('interaction_id').groupBy(group_cols).count()
    test_R_msid = interactions_test.groupBy(group_cols).count()
    
    # add track_id 
    train_R = train_R_msid.join(df_trackid, 'recording_msid').select('user_id', 'track_id', 'count')
    valid_R = valid_R_msid.join(df_trackid, 'recording_msid').select('user_id', 'track_id', 'count')
    test_R = test_R_msid.join(df_trackid, 'recording_msid').select('user_id', 'track_id', 'count')
    
    # save train
    train_R_sorted = train_R.sort('user_id', 'track_id')
    train_R_sorted.show()
    train_R_sorted.write.partitionBy("user_id").mode('overwrite').parquet(f'/user/{username}/1004-project-2023/train_inter_small_3.parquet')
    
    # create true labels for valid and save valid
    valid_true = valid_R.groupby('user_id').agg(collect_set('track_id').alias('true'))
    valid_true_sorted = valid_true.sort('user_id')
    valid_true_sorted.show()
    valid_true_sorted.write.partitionBy("user_id").mode('overwrite').parquet(f'/user/{username}/1004-project-2023/vali_inter_small_3.parquet')
    
    # create true labels for test and save test
    test_true = test_R.groupby('user_id').agg(collect_set('track_id').alias('true'))
    test_true_sorted = test_true.sort('user_id')
    test_true_sorted.show()
    test_true_sorted.write.partitionBy("user_id").mode('overwrite').parquet(f'/user/{username}/1004-project-2023/test_inter_3.parquet')

if __name__ == "__main__":
    spark = SparkSession.builder.appName('final-project').getOrCreate()
    username = getpass.getuser()
    trainvalisplit(spark, username)
    spark.stop()
