# Suicide Detection Using Machine Learning

This is an NLP model that uses machine learning algorithms to detect signs of suicidal ideation in social media posts or forum entries. The project involves collecting a dataset of text annotated with labels indicating suicidal ideation, preprocessing the data, extracting relevant features, and training various machine learning classification algorithms to accurately identify text indicative of suicidal thoughts. The goal is to create a reliable tool to identify individuals at risk and provide timely intervention.

# Dataset Description:
Dataset used is a kaggle dataset. The dataset is a collection of posts from the "SuicideWatch" and "depression" subreddits of the Reddit platform. The posts are collected using Pushshift API. All posts that were made to "SuicideWatch" from Dec 16, 2008(creation) till Jan 2, 2021, were collected while "depression" posts were collected from Jan 1, 2009, to Jan 2, 2021. All posts collected from SuicideWatch are labeled as suicide, While posts collected from the depression subreddit are labeled as depression. Non-suicide posts are collected from r/teenagers.

# Machine Learning Algorithm used:
Following are machine learning classification algorithms used for prediction:
1. Naive Bayes Classifiers
2. Random Forest
3. Decision Tree
4. Gradient Boosting
5. XG Boost
6. K-Nearest Neighbour (KNN)
