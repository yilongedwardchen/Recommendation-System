# A Collaborative-Filter Based Recommender System
### Overview
The final project for DSGA1004 - Big Data involves building and evaluating a collaborative-filter based recommender system. The project involves working with a large-scale real-world dataset and applying the tools and techniques learned in the course. The work for this project was carried out between April and May 2023.

### Dataset
The data used in this project is the ListenBrainz dataset, which comprises implicit feedback from music listening behavior, spanning several thousand users and tens of millions of songs. Each observation in the data represents a single interaction between a user and a song.

### Approach
The project began with data management and processing tasks, handling a colossal dataset of 50 million song interactions. This was done using PySpark, which greatly enhanced model efficiency and predictive accuracy. The interaction data was partitioned into training and validation samples, and the observed interaction events were aggregated into count data.

The recommendation model was built using PySpark's machine learning library. The model utilized the Alternating Least Squares (ALS) method to learn latent factor representations for users and items. Extensive hyperparameter tuning was conducted to optimize model performance on the validation set.

The popularity baseline model was implemented and evaluated first. This model achieved a Mean Average Precision at K (MAP@K) score of 0.00099. The ALS recommendation system significantly improved this score, increasing it by 40 times to 0.041.

### Extensions and Enhancements
To build on the ALS recommender system, a comprehensive analysis of LightFM's scalability and efficiency limitations was conducted. This analysis provided critical insights that informed substantial system enhancements.

A key enhancement made to the system was the integration of the ANNOY Fast Search system. This improved the quality of recommendations by 16%, resulting in a final MAP@K score of 0.048. This key improvement contributed to increased accuracy and relevance of the recommendations provided by the system.

### Results and Contributions
The project resulted in a highly efficient and accurate recommender system. The final model was capable of handling a large-scale dataset and providing highly relevant recommendations to the users. Each member of the project team contributed significantly to various stages of the project, from data processing to model optimization and system enhancements.

### Conclusion
This project offered a valuable opportunity to apply big data tools and techniques to a realistic, large-scale problem. The resulting recommender system not only achieved high performance but also provided key insights into the scalability and efficiency limitations of recommendation models. The experience gained from this project will undoubtedly prove valuable in future big data projects.
