#!/usr/bin/env python
# coding: utf-8

# <div class="alert alert-block alert-success">
#     
# # FIT5196 Task 3 in Assessment 1
# #### Student Name: Zihan Yin
# #### Student ID: 34502297
# 
# Date: 2024.08.27
# 
# Environment: Python 3.12.4
# 
# Libraries used:
# * json
# * pandas
# * numpy
# * collections
# * plotnine
#     
# </div>

# <div class="alert alert-block alert-danger">
#     
# ## Table of Contents
# 
# </div>    
# 
# [Introduction](#0) <br>
# [Step 0: Preparation](#0) <br>
# $\;\;\;\;$[0.1 Importing Libraries](#0.1) <br>
# $\;\;\;\;$[0.2 Importing Data](#0.2) <br>
# $\;\;\;\;\;\;\;\;$[0.2.1 Import Google Maps review data](#0.2.1) <br>
# $\;\;\;\;\;\;\;\;$[0.2.2 Import metadata](#0.2.2) <br>
# [Step 1: Understand the Sample Google Review Data](#1) <br>
# $\;\;\;\;$[1.1 Processing for Subsequent Analysis](#1.1) <br>
# $\;\;\;\;\;\;\;\;$[1.1.1 Data Integration & Split](#1.1.1) <br>
# $\;\;\;\;\;\;\;\;$[1.1.2 Converting Data Type & Simple Feature Engineering](#1.1.2) <br>
# $\;\;\;\;$[1.2 Summarising for `gmap_info`](#1.2) <br>
# $\;\;\;\;$[1.3 Summarising for `gmap_reviews`](#1.3) <br>
# $\;\;\;\;$[1.4 Preliminary Trends and Patterns](#1.4) <br>
# [Step 2: Understand the Metadata](#2) <br>
# $\;\;\;\;$[2.1 Metadata Overview](#2.1) <br>
# $\;\;\;\;$[2.2 Processing & Feature Engineering](#2.2) <br>
# $\;\;\;\;$[2.3 Evaluate the Usefulness](#2.3) <br>
# [Step 3: Data Analysis](#3) <br>
# $\;\;\;\;$[3.1 Analyzing from the Business Perspective](#3.1) <br>
# $\;\;\;\;$[3.2 Analyzing from the User Perspective](#3.2) <br>
# [Step 4: Summary of Meaningful Insights](#4) <br>
# 

# # Introduction

# In this task, we conduct a comprehensive exploratory data analysis (EDA) on the provided Google Review data & metadata. The goal is to uncover interesting insights that can be useful for further analysis or decision-making. The whole `.ipynb` is divided into 4 steps.

# # Step 0: Preparation

# ## 0.1 Importing Libraries

# - **json**: Python's built-in library for parsing and generating JSON data.
# 
# - **pandas**: Powerful library for data manipulation and analysis, using DataFrame structures.
# 
# - **numpy**: Library for efficient array operations and numerical computing.
# 
# - **collections.defaultdict**: A dictionary subclass that provides default values for missing keys.
# 
# - **plotnine**: A Python visualization library based on ggplot2, for creating layered statistical plots.

# In[ ]:


import json
import pandas as pd
import numpy as np
from collections import defaultdict
from plotnine import *


# ## 0.2 Importing Data

# ### 0.2.1 Import Google Maps review data

# Read `task1_020.csv`

# In[ ]:


# Read the CSV file 'task1_020.csv' into a DataFrame called 'task1_020_csv'
task1_020_csv = pd.read_csv('task1_020.csv')

# Display the first 10 rows of the 'task1_020_csv' DataFrame
task1_020_csv.head(10)


# Read `task1_020.json`

# In[ ]:


# Read the JSON file 'task1_020.json' into a DataFrame called 'task1_020_json'
# The 'orient' parameter is set to 'index' to interpret the keys of the JSON object as row (indexes
task1_020_json = pd.read_json('task1_020.json', orient='index')

# Display the first 10 rows of the 'task1_020_json' DataFrame
task1_020_json.head(10)


# ### 0.2.2 Import metadata

# In[ ]:


# Read the JSON file 'meta-California.json' into a DataFrame called 'metadata'
# The 'lines=True' parameter indicates that each line in the file is a separate JSON object
metadata = pd.read_json('meta-California.json', lines=True)

# Display the first 5 rows of the 'metadata' DataFrame
metadata.head(5)


# # Step 1: Understand the Sample Google Review Data

# ## 1.1 Processing for Subsequent Analysis

# ### 1.1.1 Data Integration & Split

# Overview of `task1_020_csv`

# In[ ]:


# Print the number of records in the 'task1_020_csv' DataFrame
print(f'The number of records: {len(task1_020_csv)}')

# Display the first 5 rows of 'task1_020_csv'
task1_020_csv.head(5)


# Overview of `task1_020_json`

# In[ ]:


# Print the number of records in the 'task1_020_json' DataFrame
print(f'The number of records: {len(task1_020_json)}')

# Display the first 5 rows of 'task1_020_json'
task1_020_json.head(5)


# Merge `task1_020_csv` and `task1_020_json` into a single dataset using gmap_id as the key.

# In[ ]:


# Merge the 'task1_020_csv' and 'task1_020_json' DataFrames on 'gmap_id' from CSV and the index from JSON
gmap_info = pd.merge(task1_020_csv, task1_020_json, left_on = 'gmap_id', right_index = True)

# Print the number of records in the merged 'gmap_info' DataFrame
print(f'The number of records: {len(gmap_info)}')

# Display the first 5 rows of 'gmap_info'
gmap_info.head(5)


# The reviews column contains multiple reviews for each `gmap_id`. We will split it into a separate DataFrame called `gmap_reviews`. Use `gmap_id`, `user_id` as the row index.

# In[ ]:


# Select 'gmap_id' and 'reviews' columns from 'gmap_info' and expand each list in 'reviews' into separate rows
gmap_reviews = gmap_info[['gmap_id', 'reviews']].explode('reviews')

# Concatenate the 'gmap_id' column and the expanded 'reviews' columns (as separate columns) into a new DataFrame
gmap_reviews = pd.concat([
        gmap_reviews['gmap_id'], 
        gmap_reviews['reviews'].apply(pd.Series)  # Apply pd.Series to expand the 'reviews' dictionary into separate columns
    ], 
    axis = 1
)

# Set the index of the DataFrame to be a multi-index using 'gmap_id' and 'user_id' for easier data access
gmap_reviews = gmap_reviews.set_index(['gmap_id', 'user_id'])

# Display the final DataFrame 'gmap_reviews'
gmap_reviews


# Remove the reviews column from `gmap_info`. Use `gmap_id` as the row index.

# In[ ]:


# Drop the 'reviews' column from the 'gmap_info' DataFrame
# Set 'gmap_id' as the row index of the DataFrame
gmap_info = (gmap_info
    .drop('reviews', axis = 1)
    .set_index('gmap_id')
)

# Display the updated 'gmap_info' DataFrame
gmap_info


# `gmap_info` and `gmap_reviews` will be the primary datasets we focus on in Step 1. Specifically:
# 
# * The row index of `gmap_info` is `gmap_id`, with each row representing the related information of a `gmap_id` (business).
# * The row index of gmap_reviews is (`gmap_id`, `user_id`), with each row representing information about a user’s review of a specific business.
# 
# The following two pieces of code ensure that the row indices of `gmap_info` and `gmap_reviews` are unique identifiers.

# In[ ]:


# Check the occurrence of each gmap_id in gmap_info using groupby
grouped = gmap_info.groupby(['gmap_id']).size()

# Check if there are any counts greater than 1, indicating duplicates
if (grouped > 1).any():
    print("Row index gmap_id is not a unique identifier. Duplicate values do exist.")
else:
    print("Row index gmap_id is a unique identifier. Duplicate values do not exist.")


# In[ ]:


# Check the occurrence of each (gmap_id, user_id) combination in gmap_reviews using groupby
grouped = gmap_reviews.groupby(['gmap_id', 'user_id']).size()

# Check if there are any counts greater than 1, indicating duplicates
if (grouped > 1).any():
    print("Hierarchical index (gmap_id, user_id) is not a unique identifier. Duplicate values do exist.")
else:
    print("Hierarchical index (gmap_id, user_id) is a unique identifier. Duplicate values do not exist.")


# `gmap_info` specific features：
# - `gmap_id`: business ID
# - `review_count`: the number of total reviews for a business.
# - `review_text_count`: the number of reviews that contains a text.
# - `response_count`: the number of responses from a business.
# - `earliest_review_date`: the earliest review date for a given business
# - `latest_review_date`: the latest review date for a given business

# In[ ]:


# Print the number of records in the 'gmap_info' DataFrame
print(f'The number of records: {len(gmap_info)}')

# Display the first 5 rows of the 'gmap_info' DataFrame
gmap_info.head(5)


# `gmap_reviews` specific features：
# - `gmap_id`:the ID of the business
# - `user_id`: ID of the reviewer.
# - `time`: the time of the review.
# - `review_rating`: rating of the business.
# - `review_text`: the english review text
# - `if_pic`: if the reviewer include pictures.
# - `pic_dim`: the dimension of pictures in a list of tuples.
# - `if_response`: if the review has a response

# In[ ]:


# Print the number of records in the 'gmap_reviews' DataFrame
print(f'The number of records: {len(gmap_reviews)}')

# Display the entire 'gmap_reviews' DataFrame
gmap_reviews


# ### 1.1.2 Converting Data Type & Simple Feature Engineering

# In[ ]:


# Display the data types of each column in the 'gmap_info' DataFrame
gmap_info.dtypes


# In[ ]:


# Convert the 'earliest_review_date' column to datetime format
gmap_info['earliest_review_date'] = pd.to_datetime(gmap_info['earliest_review_date'])

# Convert the 'latest_review_date' column to datetime format
gmap_info['latest_review_date'] = pd.to_datetime(gmap_info['latest_review_date'])


# In[ ]:


# Display the data types of each column in the 'gmap_info' DataFrame
gmap_info.dtypes


# In[ ]:


# Display the data types of each column in the 'gmap_reviews' DataFrame
gmap_reviews.dtypes


# In[ ]:


# Convert the 'time' column in 'gmap_reviews' to datetime format
gmap_reviews['time'] = pd.to_datetime(gmap_reviews['time'])


# We believe that the `if_pic` and `pic_dim` columns overlap in meaning, so we have merged them into a single column called `n_pictures`, which represents the number of pictures in a review.

# In[ ]:


# Create a new column 'n_pictures' in 'gmap_reviews' that counts the number of pictures in each review
gmap_reviews['n_pictures'] = gmap_reviews['pic_dim'].apply(lambda x: len(x))

# Drop the 'if_pic' and 'pic_dim' columns from 'gmap_reviews' as they are no longer needed
gmap_reviews = gmap_reviews.drop(['if_pic', 'pic_dim'], axis = 1)


# In[ ]:


# Display the data types of each column in the updated 'gmap_reviews' DataFrame
gmap_reviews.dtypes


# In[ ]:


# Display the entire 'gmap_reviews' DataFrame
gmap_reviews


# Check for missing values in `gmap_info` and `gmap_reviews`.

# In[ ]:


# Apply a lambda function to each column in 'gmap_info' to count the number of missing values (NaNs)
gmap_info.apply(lambda col: (col.isnull()).sum())


# In[ ]:


# Apply a lambda function to each column in 'gmap_reviews' to count the number of missing values (NaNs)
gmap_reviews.apply(lambda col: (col.isnull()).sum())


# It looks very clean, with no 'None' values. However, at the end of Task 1, to ensure the output file met the requirements, we replaced all missing values with the string 'None'. These 'None' values are present in the `review_text` column of `gmap_reviews`.

# In[ ]:


# Count the number of occurrences of the string "None" in the 'review_text' column of 'gmap_reviews'
gmap_reviews.replace("None", np.nan).apply(lambda col: (col.isnull()).sum())


# We replace all into an empty string `""`。

# In[ ]:


# Replace the string "None" with actual NaN values in the 'gmap_reviews' DataFrame
# Then, fill any NaN values with an empty string ("") in the entire DataFrame
gmap_reviews = gmap_reviews.replace("None", np.nan).fillna("")


# In[ ]:


# Apply a lambda function to each column in 'gmap_reviews' to count the number of missing values (NaNs)
gmap_reviews.apply(lambda col: (col.isnull()).sum())


# We want to generate a len_`review_text` column using the `review_text` column for further analysis. `len_review_text` represents the number of words (length) in each review.

# In[ ]:


# Create a new column 'len_review_text' that calculates the number of words in each 'review_text'
gmap_reviews['len_review_text'] = gmap_reviews['review_text'].apply(lambda x: len(x.split()))

# View the result by displaying the 'review_text' and 'len_review_text' columns for the first few records
gmap_reviews[['review_text', 'len_review_text']].head()


# ## 1.2 Summarising for `gmap_info`

# In this section, we focus on the univariate analysis of the `gmap_info` dataset. Specifically, we examine the distribution of each column.

# In[ ]:


# Print the number of records in the 'gmap_info' DataFrame
print(f'The number of records: {len(gmap_info)}')

# Display the first 5 rows of the 'gmap_info' DataFrame
gmap_info.head(5)


# The following statistics show:
# - `review_count`, `review_text_count`, and `response_count` all exhibit strong right-skewed distributions, especially `response_count`, with a median of 0.
# - The most of `latest_review_date` values are concentrated in recent years, while `earliest_review_date` values are generally distributed further in the past.
# 
# These statistical summaries are not very intuitive, so we create some visualizations to better understand the data.

# In[ ]:


# Generate descriptive statistics for the 'gmap_info' DataFrame
gmap_info.describe()


# Since `review_count`, `review_text_count`, and `response_count` are all related to the number of reviews/responses, we can overlay them in a single visualization. Below is the histogram for `review_count`, `review_text_count`, and `response_count`.

# In[ ]:


# Select the columns 'review_count', 'review_text_count', and 'response_count' from 'gmap_info'
# Reshape the DataFrame to a long format for easier plotting, with 'variable' and 'value' columns
gmap_info_for_visual = gmap_info[['review_count', 'review_text_count', 'response_count']].melt(var_name = 'variable', value_name = 'value')

# Create a frequency polygon plot to visualize the distribution of review counts, text counts, and response counts
(ggplot(gmap_info_for_visual, mapping = aes(x = 'value', color = 'variable')) +
    geom_freqpoly(bins = 50) +  # Plot the frequency polygons with 50 bins
    labs(
        title = 'Distribution of Review Counts, Text Counts, and Response Counts',  # Add a title to the plot
        x = 'Value',  # Label for the x-axis
        y = 'Count'   # Label for the y-axis
    ) +
    theme_minimal() +  # Apply a minimalistic theme to the plot
    theme(
        figure_size = (8, 5.2),  # Set the size of the figure
        panel_background = element_rect(fill = 'white'),  # Set the panel background color to white
        plot_background = element_rect(fill = 'white'),  # Set the overall plot background color to white
        plot_title = element_text(hjust = 0),  # Align the plot title to the left
        panel_grid_minor = element_line(color = 'lightgrey', size = 0.4)  # Style the minor grid lines
    )
)


# From the chart, we can observe the following points:
# - **Concentration:** All three variables are concentrated in the lower value range.
# - **Similarity between `review_count` and `review_text_count`:** The distribution shapes of the green and blue lines are similar, indicating a certain correlation between the number of text reviews and the total number of reviews.
# - **Sparsity of `response_count`:** The red line shows that the `response_count` data is significantly concentrated in the lower value range. The distribution pattern suggests that most businesses rarely respond to reviews. This may reflect a lack of attention to customer feedback in some industries.
# - **Long-Tail Distribution:** All variables exhibit a long-tail distribution, especially `review_count`. This indicates that a small number of businesses have a large number of reviews, text reviews, and even responses. We hypothesize that these businesses may have higher ratings.

# For `earliest_review_date` and `latest_review_date`, we also create an overlay histogram similar to the one above

# In[ ]:


# Convert the datetime columns ('earliest_review_date' and 'latest_review_date') to long format
gmap_info_for_visual = gmap_info[['earliest_review_date', 'latest_review_date']].melt(var_name = 'variable', value_name = 'date')

# Plot a frequency polygon for the distribution of the datetime values
(ggplot(gmap_info_for_visual, mapping = aes(x = 'date', color = 'variable')) + 
    geom_freqpoly(bins = 40) +  # Plot the frequency polygons with 40 bins
    labs(
        title = 'Distribution of Earliest and Latest Review Dates',  # Add a title to the plot
        x = 'Date',  # Label for the x-axis
        y = 'Count'  # Label for the y-axis
    ) +
    theme_minimal() +  # Apply a minimalistic theme to the plot
    theme(
        figure_size = (8, 5.2),  # Set the size of the figure
        panel_background = element_rect(fill = 'white'),  # Set the panel background color to white
        plot_background = element_rect(fill = 'white'),  # Set the overall plot background color to white
        plot_title = element_text(hjust = 0),  # Align the plot title to the left
        panel_grid_minor = element_line(color = 'lightgrey', size = 0.4)  # Style the minor grid lines
    )
)


# From the chart, we can observe:
# - The distribution of the earliest review dates is relatively dispersed, mostly concentrated between 2010 and 2020.
# - The latest review dates have increased significantly after 2020, indicating very active review activity in recent years.
# - Overall, this reflects that businesses have gained more user attention in recent years, and users have increasingly used Google Maps over the past few years.
# - This also indirectly suggests potential for the recommendation system within Google Maps (as seen later in the `relative_results` column of the metadata).

# ## 1.3 Summarising for `gmap_reviews`

# In this section, we focus on the univariate analysis of the `gmap_reviews` dataset. Specifically, we examine the distribution of each column.
# 

# In[ ]:


# Print the number of records in the 'gmap_info' DataFrame
print(f'The number of records: {len(gmap_info)}')

# Display the first 5 rows of the 'gmap_reviews' DataFrame
gmap_reviews.head(5)


# The following statistics show:
# - **review_rating**: The `review_rating` column exhibits a strong left-skewed distribution, as its 25th percentile is 4.0.
# - **n_pictures**: The vast majority of reviews have no pictures, with the 75th percentile of the `n_pictures` column being 0.
# - **len_review_text**: Similarly, most reviews have either no text or very short text. There are a few reviews with exceptionally long text.
# - **time**: Similar to the distribution of the `latest_review_time` column in `gmap_info`, most reviews were published in recent years.
# - **if_response**: The `if_response` column is not displayed here due to its data type.
# 
# As before, we create some visualizations to better understand the data.
# 

# In[ ]:


# Generate descriptive statistics for the 'gmap_reviews' DataFrame
gmap_reviews.describe()


# 
# Although `review_rating` is a numerical column, it has only five unique values. We treat it as a categorical feature and plot a bar plot alongside the `if_response` column to visualize their respective frequencies.
# 

# In[ ]:


# Ensure all data is converted to string type for consistent plotting
gmap_reviews_for_visual = (gmap_reviews[['review_rating', 'if_response']]
    .astype(str)  # Convert the selected columns to string type
    .melt(var_name = 'variable', value_name = 'value')  # Reshape the DataFrame to long format
)

# Plotting the frequency of review ratings and response presence
(ggplot(gmap_reviews_for_visual, mapping = aes(x = 'value')) +
    geom_bar(mapping = aes(fill = 'variable')) +  # Create a bar plot, with bars colored by 'variable'
    labs(
        title = 'Frequency of Review Ratings and Response Presence',  # Add a title to the plot
        x = 'Value',  # Label for the x-axis
        y = 'Count'   # Label for the y-axis
    ) +
    facet_wrap('~ variable', ncol = 1, scales = 'free_x') +  # Create separate subplots for each variable, with independent x-axis scales
    theme_minimal() +  # Apply a minimalistic theme to the plot
    theme(
        figure_size = (8, 5.2),  # Set the size of the figure
        panel_background = element_rect(fill = 'white'),  # Set the panel background color to white
        plot_background = element_rect(fill = 'white'),  # Set the overall plot background color to white
        plot_title = element_text(hjust = 0),  # Align the plot title to the left
        panel_grid_minor = element_line(color = 'lightgrey', size = 0.4),  # Style the minor grid lines
        strip_text = element_blank()  # Hide the subplot labels (facet strip texts)
    )
)


# From the chart, we can observe:
# - Most reviews did not receive a response (N), as indicated by the taller bar on the left. The number of reviews without a response is much greater than those with a response (Y).
# - Review ratings are concentrated in the higher range (4 and 5 stars), with 5-star ratings being the most frequent, close to 20,000. Low ratings (1 to 3 stars) are relatively fewer.

# We plot histograms for the `n_pictures`, `len_review_text`, and `time` columns separately to examine their distributions.

# In[ ]:


# Create a new DataFrame from 'gmap_reviews' and convert the 'time' column to the year, without modifying the original dataset
gmap_reviews_for_visual = gmap_reviews[['n_pictures', 'len_review_text', 'time']].copy()
gmap_reviews_for_visual['year'] = gmap_reviews_for_visual['time'].dt.year

# Reshape the DataFrame to long format, using the newly created 'year' column and keeping the original 'time' column unchanged
gmap_reviews_for_visual = (gmap_reviews_for_visual[['n_pictures', 'len_review_text', 'year']]
    .melt(var_name = 'variable', value_name = 'value')
)

# Plotting the histogram for the distribution of review text length, number of pictures, and review time (by year)
(ggplot(gmap_reviews_for_visual, mapping = aes(x = 'value')) +
    geom_histogram(mapping = aes(fill = 'variable'), bins = 20) +  # Create a histogram with 20 bins, colored by variable
    labs(
        title = 'Distribution of Length of Review Text, #Pictures, and Review Time',  # Add a title to the plot
        x = 'Value',  # Label for the x-axis
        y = 'Count'   # Label for the y-axis
    ) +
    facet_wrap('~ variable', scales = 'free', ncol = 1) +  # Create separate subplots for each variable, with independent scales
    scale_fill_brewer(palette = 'Set2', type = 'qual') +  # Use a qualitative color palette for fill colors
    theme_minimal() +  # Apply a minimalistic theme to the plot
    theme(
        figure_size = (8, 5.2),  # Set the size of the figure
        panel_background = element_rect(fill = 'white'),  # Set the panel background color to white
        plot_background = element_rect(fill = 'white'),  # Set the overall plot background color to white
        plot_title = element_text(hjust = 0),  # Align the plot title to the left
        panel_grid_minor = element_line(color = 'lightgrey', size = 0.4),  # Style the minor grid lines
        strip_text = element_blank()  # Hide the subplot labels (facet strip texts)
    )
)


# From the charts, we can see:
# - **len_review_text**: The majority of review texts are short, concentrated between 0 and 100 words. The number of reviews decreases rapidly as text length increases. The distribution is right-skewed, with only a small number of reviews having long texts (over 200 words).
# - **n_pictures**: Most reviews do not contain any pictures, which is evident from the tall bar at 0. Only a small number of reviews include a few pictures, with almost none containing more than 5 pictures.
# - **time**: Here, we extracted the year from the `time` column. The number of reviews increases gradually over the years, with a significant rise after 2015, peaking between 2018 and 2020. This pattern closely mirrors the distribution of `lastest_review_time` because it can be considered as a mixed distribution of `earliest_review_time` and `lastest_review_time`.

# ## 1.4 Preliminary Trends and Patterns

# In[ ]:


# Display the first 5 rows of the 'gmap_info' DataFrame
gmap_info.head(5)


# In[ ]:


# Display the first 5 rows of the 'gmap_reviews' DataFrame
gmap_reviews.head(5)


# We are particularly interested in exploring the relationship between the `review_rating` column and other features. Intuitively, we believe it should be related to the number of pictures and the word count of the reviews. To investigate this, we plotted the following two box plots.

# In[ ]:


# Filter the 'gmap_reviews' DataFrame to include only reviews with a text length of less than 600 words
gmap_reviews_for_visual = gmap_reviews[gmap_reviews['len_review_text'] < 600]

# Create a boxplot to visualize the distribution of review text length across different ratings
(ggplot(gmap_reviews_for_visual, mapping = aes(x = 'factor(review_rating)', y = 'len_review_text')) +
    geom_boxplot() +  # Add a boxplot layer to the plot
    labs(
        title = 'Distribution of Review Text Length Across Ratings',  # Add a title to the plot
        x = 'Review Rating',  # Label for the x-axis
        y = 'Review Text Length'  # Label for the y-axis
    ) +
    theme_minimal() +  # Apply a minimalistic theme to the plot
    theme(
        figure_size = (8, 5.2),  # Set the size of the figure
        panel_background = element_rect(fill = 'white'),  # Set the panel background color to white
        plot_background = element_rect(fill = 'white'),  # Set the overall plot background color to white
        plot_title = element_text(hjust = 0),  # Align the plot title to the left
        panel_grid_minor = element_line(color = 'lightgrey', size = 0.4),  # Style the minor grid lines
    )
)


# From the first box plot above, we can observe:
# - Regardless of the rating, the review text length is generally concentrated in a short range.
# - Nevertheless, it can be noted that reviews with a rating of 1.0 tend to have relatively longer review texts.

# In[ ]:


# Filter the 'gmap_reviews' DataFrame to include only reviews with fewer than 20 pictures
gmap_reviews_for_visual = gmap_reviews[gmap_reviews['n_pictures'] < 20]

# Create a boxplot to visualize the distribution of the number of pictures across different ratings
(ggplot(gmap_reviews_for_visual, mapping = aes(x = 'factor(review_rating)', y = 'n_pictures')) +
    geom_boxplot() +  # Add a boxplot layer to the plot
    labs(
        title = 'Distribution of Number of Pictures Across Ratings',  # Add a title to the plot
        x = 'Review Rating',  # Label for the x-axis
        y = 'Number of Pictures'  # Label for the y-axis
    ) +
    theme_minimal() +  # Apply a minimalistic theme to the plot
    theme(
        figure_size = (8, 5.2),  # Set the size of the figure
        panel_background = element_rect(fill = 'white'),  # Set the panel background color to white
        plot_background = element_rect(fill = 'white'),  # Set the overall plot background color to white
        plot_title = element_text(hjust = 0),  # Align the plot title to the left
        panel_grid_minor = element_line(color = 'lightgrey', size = 0.4),  # Style the minor grid lines
    )
)


# From the second box plot above, we can see that most reviews contain very few pictures, to the extent that both Q1 and Q3 are at 0 (hence the rectangle of the box plot is not visible). This plot does not reveal any clear relationship between `review_rating` and the number of pictures.

# If we consider the review text length and the number of pictures together, would the `review_rating` show any changes? Given that reviews with a rating of 5 are significantly more frequent than those with the other four ratings, we can combine ratings 1 to 4 into one category. In the scatter plot below, red and blue points represent ratings 1-4 and 5, respectively.

# In[ ]:


# Define a function to categorize the review ratings
def categorize_rating(rating):
    if rating in [1, 2, 3, 4]:
        return '1-4'  # Group ratings 1 to 4 together
    elif rating == 5:
        return '5'  # Separate category for rating 5

# Filter the 'gmap_reviews' DataFrame for reviews with fewer than 20 pictures and less than 600 words
gmap_reviews_for_visual = gmap_reviews[(gmap_reviews['n_pictures'] < 20) & (gmap_reviews['len_review_text'] < 600)]

# Apply the 'categorize_rating' function to create a new 'rating_category' column
gmap_reviews_for_visual['rating_category'] = gmap_reviews_for_visual['review_rating'].apply(categorize_rating)

# Create a scatter plot to visualize the relationship between review text length and number of pictures by rating category
(ggplot(gmap_reviews_for_visual, mapping = aes(x = 'len_review_text', y = 'n_pictures', colour = 'rating_category')) +
    geom_point(alpha = 0.6) +  # Add points to the plot with transparency for overlap
    labs(
        title = 'Relationship Between Review Text Length and Number of Pictures by Rating Category',  # Add a title to the plot
        x = 'Length of Review Text',  # Label for the x-axis
        y = 'Number of Pictures',  # Label for the y-axis
        colour = 'Rating Category'  # Label for the legend (colour)
    ) +
    theme_minimal() +  # Apply a minimalistic theme to the plot
    theme(
        figure_size = (10, 5.2),  # Set the size of the figure
        panel_background = element_rect(fill = 'white'),  # Set the panel background color to white
        plot_background = element_rect(fill = 'white'),  # Set the overall plot background color to white
        plot_title = element_text(hjust = 0),  # Align the plot title to the left
        panel_grid_minor = element_line(color = 'lightgrey', size = 0.4),  # Style the minor grid lines
        strip_text = element_blank()  # Hide the subplot labels (facet strip texts)
    )
)


# Trends observed in the above plot:
# 
# - **Number of Pictures**: Most reviews contain a low number of pictures, concentrated between 0 and 5. Reviews with a rating of 5 seem to be more concentrated within this lower range of picture numbers.
# 
# - **Text Length**: Most reviews also have shorter text lengths, concentrated between 0 and 200 words. As text length increases, there is no significant increase in the number of accompanying pictures. Reviews with a rating of 5 seem to be more concentrated in shorter texts.
# 
# These trends suggest that higher-rated reviews (those with a rating of 5) tend to contain fewer pictures and shorter text lengths. This is somewhat expected, as some businesses might engage in practices where fake accounts are used to artificially inflate their ratings.

# We are also interested in whether there is a relationship between review rating and the hour of the day. We extracted the hour of the day from the `time` column.

# In[ ]:


# Create a copy of the 'gmap_reviews' DataFrame for visualization purposes
gmap_reviews_for_visual = gmap_reviews.copy()

# Extract the hour from the 'time' column and create a new 'hour' column
gmap_reviews_for_visual['hour'] = gmap_reviews_for_visual['time'].dt.hour 

# Create a plot to visualize the distribution of review ratings across different hours of the day
(ggplot(gmap_reviews_for_visual, mapping=aes(x='hour', y='review_rating')) +
    geom_count() +  # Add a geom_count layer to show the frequency of each combination of hour and review rating
    labs(
        title='Distribution of Review Ratings Across Different Hours of the Day',  # Add a title to the plot
        x='Hour of the Day',  # Label for the x-axis
        y='Review Rating'  # Label for the y-axis
    ) +
    theme_minimal() +  # Apply a minimalistic theme to the plot
    theme(
        figure_size=(8, 5.2),  # Set the size of the figure
        panel_background=element_rect(fill='white'),  # Set the panel background color to white
        plot_background=element_rect(fill='white'),  # Set the overall plot background color to white
        plot_title=element_text(hjust=0),  # Align the plot title to the left
        panel_grid_minor=element_line(color='lightgrey', size=0.4)  # Style the minor grid lines
    )
)


# In the above chart, the size of the dots represents the number of reviews given a specific rating at a particular hour of the day. The larger the dot, the higher the number of reviews. From the chart, we can observe:
# 
# - Low ratings of 1-3 are distributed throughout the day, with slightly more reviews at certain hours, but overall, there is no clear concentration trend.
# 
# - Certain hours, such as 1 AM and 9 PM, show a significantly higher number of reviews than other times, indicating that users are more active during these periods.
# 
# Overall, while we cannot draw a strong correlation between review rating and the hour of the day from this chart, we do observe significant differences in the number of reviews given at different hours of the day. The following bar plot was created to confirm our findings:
# 

# In[ ]:


# Create a bar plot to visualize the frequency of reviews throughout the day
(ggplot(gmap_reviews_for_visual, mapping=aes(x='hour')) +
    geom_bar(fill='dodgerblue') +  # Create a bar plot with bars filled in 'dodgerblue' color
    labs(
        title='Frequency of Reviews Throughout the Day',  # Add a title to the plot
        x='Hour of the Day',  # Label for the x-axis
        y='Review Count'  # Label for the y-axis
    ) +
    theme_minimal() +  # Apply a minimalistic theme to the plot
    theme(
        figure_size=(8, 5.2),  # Set the size of the figure
        panel_background=element_rect(fill='white'),  # Set the panel background color to white
        plot_background=element_rect(fill='white'),  # Set the overall plot background color to white
        plot_title=element_text(hjust=0),  # Align the plot title to the left
        panel_grid_minor=element_line(color='lightgrey', size=0.4)  # Style the minor grid lines
    )
)


# From the above bar plot, we can observe:
# 
# - **Midnight Peak:** Between 12 AM and 5 AM, there is a increase in the number of reviews. This might be because many users tend to review their experiences at the end of the day.
# 
# - **Morning and Noon Low:** From 6 AM to 12 PM, the number of reviews sharply decreases. This time period is typically when people are starting their day, busy with work or other daily activities, so the number of reviews is relatively low.
# 
# - **Afternoon to Evening Recovery:** After 3 PM, the number of reviews starts to increase, peaking again around 8 PM. This trend reflects users spending time reviewing their experiences after finishing their day’s work.
# 
# Overall, users tend to leave reviews outside of work hours, which aligns with typical daily routines.

# # Step 2: Understand the Metadata

# ## 2.1 Metadata Overview

# Metadata feature description：
# 
# * name - name of the business
# 
# * address - address of the business
# 
# * gmap_id - ID of the business
# 
# * description - description of the business
# 
# * latitude - latitude of the business
# 
# * longitude - longitude of the business
# 
# * category - category of the business
# 
# * avg_rating - average rating of the business
# 
# * num_of_reviews - number of reviews
# 
# * price - price of the business
# 
# * hours - open hours
# 
# * MISC - MISC information
# 
# * state - the current status of the business (e.g., permanently closed)
# 
# * relative_results - relative businesses recommended by Google
# 
# * url - URL of the business

# In[ ]:


# Print the number of records in the 'metadata' DataFrame
print(f'The number of records: {len(metadata)}')

# Display the first 5 rows of the 'metadata' DataFrame
metadata.head(5)


# Set gmap_id as the row index for the metadata DataFrame.

# In[ ]:


# Set 'gmap_id' as the row index for the 'metadata' DataFrame
metadata = metadata.set_index('gmap_id')


# We find there exists duplicated rows in metadata and choose to remove them.

# In[ ]:


metadata.index.duplicated().sum()


# In[ ]:


metadata = metadata[~metadata.index.duplicated(keep='first')]


# In[ ]:


metadata.index.duplicated().sum()


# check the data types of each column in the `metadata`

# In[ ]:


metadata.dtypes


# Check the number of missing values in each column.

# In[ ]:


metadata.apply(lambda col: (col.isnull()).sum())


# ## 2.2 Processing & Featrue Engineering

# Generate a location column using longitude and latitude information.

# In[ ]:


# Combine the 'latitude' and 'longitude' columns into a single 'location' column as a list [latitude, longitude]
metadata['location'] = metadata.apply(lambda row: [row['latitude'], row['longitude']], axis=1)

# Drop the original 'latitude' and 'longitude' columns since their data is now in the 'location' column
metadata = metadata.drop(['latitude', 'longitude'], axis=1)

# Display the first 5 rows of the updated 'metadata' DataFrame
metadata.head(5)


# Extract the city information from the address column and create a new city column.

# In[ ]:


# For each row in the 'address' column, check if the value is not None or NaN
# If valid, split the address by commas and extract the second last element as the city name
# If the address is None or has fewer than 2 elements after splitting, return "Unknown City"
metadata['city'] = metadata['address'].apply(
    lambda x: x.split(',')[-2].strip() 
        if pd.notnull(x) and len(x.split(',')) >= 2 
        else "Unknown City"
)

# Drop the original 'address' column as its data is now in the 'city' column
metadata = metadata.drop('address', axis=1)

# Display the first 5 rows of the updated 'metadata' DataFrame
metadata.head(5)


# Remove the columns we are not interested in.

# In[ ]:


# Remove the 'description' and 'url' columns from the 'metadata' DataFrame
metadata = metadata.drop(['description', 'url'], axis=1)

# Display the first 5 rows of the updated 'metadata' DataFrame
metadata.head(5)


# Extract the first category from the category column and create a new primary_category column.

# In[ ]:


# Extract the first category from the 'category' column and create a new 'primary_category' column
# If 'category' is a list, extract the first element; otherwise, set it as "Unknown Category"
metadata['primary_category'] = metadata['category'].apply(lambda x: x[0] if isinstance(x, list) else "Unknown Category")

# Drop the original 'category' column as its data is now in the 'primary_category' column
metadata = metadata.drop('category', axis=1)

# Display the first 5 rows of the updated 'metadata' DataFrame
metadata.head(5)


# Check the unique values in the price column.

# In[ ]:


print(metadata['price'].unique())


# Extract information from the price column to create a price_level column.

# In[ ]:


# Define a mapping to convert price symbols into price levels
price_mapping = {
    '$': 'Low',
    '$$': 'Medium',
    '$$$': 'High',
    '$$$$': 'Very High',
    '₩': 'Low',
    '₩₩': 'Medium',
    '₩₩₩': 'High',
    '₩₩₩₩': 'Very High',
    None: 'Unknown Price Level',  # Handle None values and NaN values
}

# Apply the mapping to the 'price' column to create the 'price_level' column
metadata['price_level'] = metadata['price'].map(price_mapping)

# Drop the original 'price' column as its data is now in 'price_level'
metadata = metadata.drop('price', axis=1)

# Print the count of each unique value in the 'price_level' column
print(metadata['price_level'].value_counts())

# Display the first 5 rows of the updated 'metadata' DataFrame
metadata.head(5)


# We extracted the number of operating days per week into the `operating_days` column and the average daily operating hours into the `average_daily_hours` column from the hours column. There are many missing values in the hours column, which results in an `operating_days` value of 0. We consider such businesses as permanently closed.

# In[ ]:


metadata['operating_days'] = metadata['hours'].apply(
    lambda x: len([day for day in x if "Closed" not in day[1]]) if isinstance(x, list) else 0
)


# In[ ]:


from datetime import datetime, timedelta

# Function to parse time strings into datetime objects
def parse_time(time_str):
    time_str = time_str.strip()
    
    # Handle the case where the business is open 24 hours
    if "Open 24 hours" in time_str:
        return "24_hours"
    
    # Handle cases where the time is given as just an hour, e.g., "3"
    if time_str.isdigit() and len(time_str) <= 2:
        return datetime.strptime(time_str + "PM", "%I%p")
    
    # Different time formats to try parsing
    time_formats = ["%I:%M%p", "%I%p", "%H:%M", "%I:%M"]
    
    for fmt in time_formats:
        try:
            return datetime.strptime(time_str, fmt)
        except ValueError:
            continue

# Function to calculate average daily operating hours
def calculate_daily_hours(hours):
    total_hours = 0
    operating_days = 0
    
    for day, time_range in hours:
        if "Closed" not in time_range:
            if "Open 24 hours" in time_range:
                hours_open = 24
            else:
                # Parse open and close times
                open_time, close_time = [parse_time(t) for t in time_range.split('–')]
                if open_time == "24_hours":
                    hours_open = 24
                else:
                    hours_open = (close_time - open_time).seconds / 3600
            total_hours += hours_open
            operating_days += 1
    
    # Calculate average daily hours, return 0 if no operating days
    return total_hours / operating_days if operating_days > 0 else 0

# Apply the calculate_daily_hours function to each row in the 'hours' column
metadata['average_daily_hours'] = metadata['hours'].apply(
    lambda x: calculate_daily_hours(x) if isinstance(x, list) else 0
)


# Remove the hours column, which has already been used. Also, remove the state column, as we consider it to have no value for feature engineering.

# In[ ]:


metadata = metadata.drop(['hours', 'state'], axis = 1)
metadata.head(5)


# The MISC column contains various service types. Under each service type, there are further detailed sub-services. As shown below:

# In[ ]:


# Initialize an empty set to store all unique keys
all_keys = set()

# Iterate through the 'MISC' column and extract keys from each dictionary
for misc in metadata['MISC'].dropna():
    for key in misc.keys():
        all_keys.add(key)

# Print all the unique keys found
print(all_keys)


# We consider the two most important service types to be 'Service options' and 'Accessibility'. We count their sub-services separately and assign these counts to the `service_options_count` and `accessibility_count columns`, respectively. Additionally, we record the total number of all `sub-services` in the MISC column as the `misc_count` column.

# In[ ]:


# Define a function to extract the count of specific features from the 'MISC' column
def count_misc_features(misc, feature_key):
    if pd.notnull(misc) and feature_key in misc:
        return len(misc[feature_key])  # Return the count of sub-services under the specified feature key
    return 0  # Return 0 if the feature key is not present or if the 'MISC' entry is null

# Calculate the count of 'Service options' and 'Accessibility' sub-services
metadata['service_options_count'] = metadata['MISC'].apply(lambda x: count_misc_features(x, 'Service options'))
metadata['accessibility_count'] = metadata['MISC'].apply(lambda x: count_misc_features(x, 'Accessibility'))

# Calculate the total count of all sub-services in the 'MISC' column
metadata['misc_count'] = metadata['MISC'].apply(lambda x: sum(len(v) for v in x.values()) if pd.notnull(x) else 0)

# Display the first few rows of the updated 'metadata' DataFrame
metadata.head()


# Calculate the number of times each business was recommended based on the `relative_results` column, and store this as the `times_recommended` column.

# In[ ]:


# Create a defaultdict to store the recommendation count for each business
recommendation_count = defaultdict(int)

# Iterate over the entire metadata DataFrame, checking the 'relative_results' column
for idx, row in metadata.iterrows():
    if isinstance(row['relative_results'], list):  # Check if 'relative_results' is a list
        for related_id in row['relative_results']:  # Iterate over the list of related business IDs
            recommendation_count[related_id] += 1  # Increment the recommendation count for each related business

# Convert the recommendation counts to a Series and merge it with the metadata DataFrame
metadata['times_recommended'] = metadata.index.map(recommendation_count.get).fillna(0).astype(int)

# Drop the 'relative_results' column as it has been processed
metadata = metadata.drop('relative_results', axis=1)

# Display the first 5 rows of the updated 'metadata' DataFrame
metadata.head(5)


# ## 2.3 Evaluate the Usefulness

# After performing feature engineering, the metadata dataset now includes a variety of features that provide additional context and information about the businesses. Below is a summary of each feature, along with its meaning:
# 
# - **name**: The name of the business.
# 
# - **avg_rating**: The average rating of the business.
# 
# - **num_of_reviews**: The total number of reviews the business has received.
# 
# - **MISC**: A dictionary containing miscellaneous information about the business, such as service options, accessibility, offerings, amenities, etc.
# 
# - **location**: A list containing the latitude and longitude of the business.
# 
# - **city**: The city where the business is located.
# 
# - **primary_category**: The primary category of the business.
# 
# - **price_level**: A categorical feature indicats the pricing level of the business, ranging from 'Low' to 'Very High'. 
# 
# - **operating_days**: The number of days per week the business is open.
# 
# - **average_daily_hours**: The average number of hours the business is open per day.
# 
# - **service_options_count**: The number of service options available at the business
# 
# - **accessibility_count**: The number of accessibility features available at the business
# 
# - **misc_count**: The total number of miscellaneous attributes (like offerings, amenities) available for the business.
# 
# - **times_recommended**: The number of times the business has been recommended by Google relative to other businesses.
# 

# In[ ]:


# Display the first 5 rows of the 'metadata' DataFrame
metadata.head(5)


# The metadata dataset provides contextual information that the main dataset lacks, significantly enhancing the analysis of the main dataset. Specifically:
# 
# - **Enhanced Feature Set**: The metadata adds numerical and categorical data such as `avg_rating`, `num_of_reviews`, `primary_category`, and `price_level`. These features offer deeper business insights, helping in trend analysis and rating prediction.
# 
# - **Business Location Information**: The inclusion of `location` enables spatial analysis, revealing geographical patterns such as clusters of similar businesses or areas with high customer satisfaction.
# 
# - **Business Operation Information**: Features like `operating_days` and `average_daily_hours` provide insights into business operations. By correlating these with customer satisfaction, number of reviews, and other performance indicators, we can understand the impact of operations on success.
# 
# - **Business Service Information**: Attributes like `accessibility_count` (count of accessibility features) and `service_options_count` (count of service options) provide information on customer service aspects of the business. This can be analyzed in relation to customer satisfaction.
# 
# - **Recommendation System Analysis**: The `times_recommended` feature indicates how frequently a business is recommended by Google, which may reflect its prominence in the market. Analyzing this alongside customer ratings can help understand the effectiveness of the recommendation system.

# Based on the above description of the metadata, we have decided to incorporate it into our analysis. By integrating the metadata, we can obtain more comprehensive information about the businesses, including their operating models, geographical locations, and customer evaluations. This will help us gain a deeper understanding of how businesses perform in the market and what customer preferences are.

# # Step 3: Data Analysis

# Let's focus on metadata and continue our analysis of the `gmap_info` and `gmap_reviews` (sample datasets) with its help. 
# 
# We'll merge metadata with `gmap_info` to enhance the information in `gmap_info`. Specifically, we'll perform a left join of metadata onto `gmap_info`.

# In[ ]:


gmap_info = pd.merge(gmap_info, metadata, how='left', left_index=True, right_index=True)

# Print the number of records in the merged 'gmap_info' DataFrame
print(f'The number of records: {len(gmap_info)}')

# Display the first 5 rows of 'gmap_info'
gmap_info.head(5)


# Next, we'll perform a left join of the `gmap_info`, which now includes metadata information, onto gmap_reviews.

# In[ ]:


# Perform a left join of gmap_info onto gmap_reviews using 'gmap_id' as the key
# The left join is done after resetting the index of gmap_reviews and using 'gmap_id' as the joining key.
# The resulting DataFrame is then set to have a multi-level index with 'gmap_id' and 'user_id'.
gmap_reviews = (pd.merge(gmap_reviews.reset_index(), gmap_info, how='left', left_on='gmap_id', right_index=True)
    .set_index(['gmap_id', 'user_id'])
)

# Print the number of records in the merged DataFrame
print(f'The number of records: {len(gmap_reviews)}')

# Display the first 5 rows of the merged DataFrame
gmap_reviews.head(5)


# Let's review the information we have:
# 
# - metadata: Focuses on business analysis and includes all businesses in the state of California.
# - gmap_info: Focuses on business analysis and includes all businesses in the sample data.
# - gmap_reviews: Focuses on user analysis and includes all users in the sample data (specifically, the reviews published by users).

# Let's start by analyzing at the business level.。

# ## 3.1 Analyzing from the Business Perspective

# We are curious about the relationship between a business's operating model (work intensity) and user behavior. Let's take a look at this boxplot, which shows the relationship between the number of operating days per week and the average operating hours per day.

# In[ ]:


# Create copies of gmap_info and metadata datasets
gmap_info_temp = gmap_info.copy()
metadata_temp = metadata.copy()

# Add a 'source' column to each dataset to identify the source of data
gmap_info_temp['source'] = 'gmap_info'
metadata_temp['source'] = 'metadata'

# Concatenate the two datasets, selecting only 'operating_days', 'average_daily_hours', and 'source' columns
combined_data = pd.concat([gmap_info_temp[['operating_days', 'average_daily_hours', 'source']],
                           metadata_temp[['operating_days', 'average_daily_hours', 'source']]])

# Create a boxplot to show the relationship between the number of operating days per week and average daily operating hours
(ggplot(combined_data, aes(x='factor(operating_days)', y='average_daily_hours', colour='source')) +
    geom_boxplot() +
    labs(
        title='Relationship Between Weekly Operating Days & Daily Operating Hour',  # Set the plot title
        x='Number of Weekly Operating Days',  # Label for the x-axis
        y='Average Daily Operating Hour'  # Label for the y-axis
    ) +
    facet_wrap('~source', ncol=1) +  # Use facet_wrap to display separate plots for each 'source'
    theme_minimal() +
    theme(
        figure_size=(9, 6),  # Set the figure size
        panel_background=element_rect(fill='white'),  # Set the panel background color to white
        plot_background=element_rect(fill='white'),  # Set the plot background color to white
        plot_title=element_text(hjust=0),  # Align the plot title to the left
        panel_grid_minor=element_line(color='lightgrey', size=0.4),  # Set minor grid lines
        strip_text=element_blank()  # Hide the subplot labels (facet strip texts)
    ) 
)


# Let's create separate plots for the relationship between the number of weekly operating days and the average daily operating hours using both the `gmap_info` and `metadata` datasets.
# 
# From the first plot (using `gmap_info`), we can observe the following:
# - As the number of weekly operating days increases, the average daily operating hours of businesses also increase. This suggests that businesses with longer operating hours tend to be open more days per week.
# - In the sample data, there are almost no businesses operating less than 3 days per week (as such businesses would likely be permanently closed).
# - Businesses operating 4 to 7 days per week show a significant increase in average daily operating hours, particularly for those operating 7 days, where the distribution of average daily operating hours is wider.
# 
# The main purpose of creating the second plot (using `metadata`) is to see if the businesses in the sample data are representative of all businesses in California. From the plot, we can see that the trends are quite consistent between the two datasets, indicating that the businesses in our sample data are indeed representative.

# The plot mainly indicates that businesses tend to choose longer operating hours if they want to operate more days per week, thereby increasing customer accessibility and service flexibility.
# 
# We are curious about the effectiveness of these operating models. Specifically, we want to know if such operating models have increased customer foot traffic. **We can approximate a business's foot traffic by using the review_count as a measure of the number of customers visiting the business.**

# In[ ]:


# Create a temporary grouping of review_count for visualization purposes
gmap_info_for_visual = gmap_info.copy()
gmap_info_for_visual['review_count_group'] = pd.qcut(gmap_info_for_visual['review_count'], q=4, labels=["Low", "Medium", "High", "Very High"])

# Create a boxplot to show the relationship between operating days, average daily hours, and review count group
(ggplot(gmap_info_for_visual, aes(x='factor(operating_days)', y='average_daily_hours', colour='review_count_group')) +
    geom_boxplot() +
    labs(
        title='Relationship Between Operating Hours, Days and Review Count Group',  # Set the plot title
        x='Average Daily Operating Hour',  # Label for the x-axis
        y='Number of Weekly Operating Days'  # Label for the y-axis
    ) +
    facet_wrap('~review_count_group') +  # Use facet_wrap to display separate plots for each review_count_group
    scale_color_brewer(palette='Set2', type='qual') +  # Use a color palette for better distinction between groups
    theme_minimal() +
    theme(
        figure_size=(10, 6),  # Set the figure size
        panel_background=element_rect(fill='white'),  # Set the panel background color to white
        plot_background=element_rect(fill='white'),  # Set the plot background color to white
        plot_title=element_text(hjust=0),  # Align the plot title to the left
        panel_grid_minor=element_line(color='lightgrey', size=0.4),  # Set minor grid lines
        strip_text=element_blank()  # Hide the subplot labels (facet strip texts)
    )
)


# The above plot is a faceted boxplot that displays the relationship between the number of weekly operating days and the average daily operating hours, grouped by the number of reviews a business has received. From the plot, we observe the following:
# 
# - In the Very High group, we do see some effect of this operating model, where businesses tend to have higher operating days and longer operating hours.
# - However, overall, there is no significant difference in the trend of operating days and hours across the four levels of customer foot traffic.
# 
# This plot suggests that the business model of extending operating hours to increase customer visits has not significantly boosted foot traffic. Customer traffic is likely influenced by other factors, such as the business's location, service quality, marketing strategies, and more, rather than just the length of operating hours and the number of operating days.

# Let's further investigate whether the city in which a business is located and its price range have an impact on customer foot traffic.

# In[ ]:


# Aggregate the data to calculate the total number of reviews for each city
city_review_count = metadata.groupby('city')['num_of_reviews'].sum()

# Select the top 10 cities with the highest total review counts
top_cities = city_review_count.nlargest(10).index

# Filter the data to keep only the records from the top 10 cities
filtered_data = metadata[metadata['city'].isin(top_cities)]

# Apply a logarithmic transformation to the number of reviews to normalize the distribution
filtered_data['log_num_of_reviews'] = np.log1p(filtered_data['num_of_reviews'])

# Set the 'price_level' column as an ordered categorical variable, defining the order
price_level_order = ['Unknown Price Level', 'Low', 'Medium', 'High', 'Very High']
filtered_data['price_level'] = pd.Categorical(filtered_data['price_level'], categories=price_level_order, ordered=True)

# Create a boxplot to visualize the log-transformed review count distribution across the top 10 cities, faceted by price level
(ggplot(filtered_data, aes(x='city', y='log_num_of_reviews', color='price_level')) +
    geom_boxplot() +
    labs(
        title='Log-Transformed Review Count Distribution Across Top 10 Cities',  # Set the plot title
        x='City',  # Label for the x-axis
        y='Log of Review Count'  # Label for the y-axis
    ) +
    facet_wrap('~price_level') +  # Facet the plot by price level
    theme(
        figure_size=(10, 6),  # Set the figure size
        panel_background=element_rect(fill='white'),  # Set the panel background color to white
        plot_background=element_rect(fill='white'),  # Set the plot background color to white
        plot_title=element_text(hjust=0),  # Align the plot title to the left
        panel_grid_minor=element_line(color='lightgrey', size=0.4),  # Set minor grid lines
        axis_text_x=element_text(rotation=90, hjust=1),  # Rotate x-axis labels to prevent overlap
        legend_position="none"  # Hide the legend
    )
)


# This plot shows the distribution of review counts in the top 10 cities, with businesses categorized by different price ranges. By applying a logarithmic transformation to the review counts, we can more clearly observe the impact of a business's city and price range on customer foot traffic (approximated by review counts). From the plot, we observe the following:
# 
# - **Unknown Price Level, Low & Medium Price Level**: For businesses in these three price ranges, the distribution of review counts across different cities is relatively even, with no significant differences observed.
# - **High & Very High Price Level**: For businesses in the high and very high price ranges, there is a noticeable variation in the distribution of review counts. Particularly in the Very High price range, businesses, although fewer in number, stand out in certain cities like Anaheim, possibly indicating a stronger market demand for high-priced businesses in these locations.
# 
# Overall, there are clear differences in customer foot traffic distribution among businesses in different cities, especially in the higher price ranges. The impact of price range on customer foot traffic is also significant (e.g., location and price range). Depending on the price range and city, businesses may consider adjusting their pricing strategies to better meet the demands of their target market.

# ## 3.2 Analyzing from the User Perspective

# We are very interested in the number of times a business has been recommended on Google Maps (`times_recommended`). Let's first take a look at its distribution.

# In[ ]:


# Plotting the histogram for the distribution of review text length, number of pictures, and review time (by year)
(ggplot(gmap_reviews, mapping=aes(x='times_recommended')) +
    geom_histogram(bins=22, fill='skyblue') +  # Create a histogram with 20 bins, colored by variable
    labs(
        title='Distribution of Times Business Being Recommended',  # Add a title to the plot
        x='Value',  # Label for the x-axis
        y='Count'   # Label for the y-axis
    ) +
    theme_minimal() +  # Apply a minimalistic theme to the plot
    theme(
        figure_size=(8, 5.2),  # Set the size of the figure
        panel_background=element_rect(fill='white'),  # Set the panel background color to white
        plot_background=element_rect(fill='white'),  # Set the overall plot background color to white
        plot_title=element_text(hjust=0),  # Align the plot title to the left
        panel_grid_minor=element_line(color='lightgrey', size=0.4)  # Style the minor grid lines
    )
)


# From the plot, we can observe the following:
# 
# - The majority of businesses have been recommended fewer than 5 times. These businesses have been recommended to some extent, but not frequently.
# - As the number of recommendations increases, the number of businesses gradually decreases. Very few businesses have been recommended more than 10 times, indicating that among the recommended businesses, only a small number receive frequent recommendations.
# 
# We can see that most businesses have low exposure in the recommendation system, with only a few enjoying a high number of recommendations.

# Next, we'll analyze the relationship between users and the Google Maps recommendation system, focusing on user behavior and influence. Specifically, we'll:
# 
# - Calculate the number of reviews each user has posted to identify "high-frequency reviewers" and then analyze whether these users tend to give higher or lower ratings.
# - Calculate the total number of reviews each user has posted and the corresponding times the businesses they reviewed have been recommended. This will help us identify which users' reviews have a significant impact on the businesses' recommendation frequency in the system.

# Let's start by identifying those users who frequently post reviews, i.e., the "high-frequency reviewers."

# In[ ]:


# Calculate the number of reviews each user has posted
user_review_counts = gmap_reviews.groupby('user_id').size().reset_index(name='review_count')

# Define the threshold for high-frequency reviewers, e.g., users with more than 10 reviews
high_frequency_threshold = 10
high_frequency_users = user_review_counts[user_review_counts['review_count'] > high_frequency_threshold]

# Print the number of high-frequency reviewers
print(f"Number of high-frequency reviewers: {high_frequency_users.shape[0]}")


# In[ ]:


# Define the threshold for high-frequency reviewers, e.g., users with more than 2 reviews
high_frequency_threshold = 2
high_frequency_users = user_review_counts[user_review_counts['review_count'] > high_frequency_threshold]

# Print the number of high-frequency reviewers
print(f"Number of high-frequency reviewers: {high_frequency_users.shape[0]}")


# The above finding is quite disappointing for us. If we set the `high_frequency_threshold` to 2, we can only identify 9 users in total. If we set the threshold to 5, we can't find a single user. This is far below our expectations.

# Let's check the number of rows in `gmap_reviews` and the number of unique users.

# In[ ]:


# Calculate the number of unique users
num_users = gmap_reviews.index.get_level_values('user_id').nunique()

# Print the number of reviews (rows) in gmap_reviews
print(f'Number of reviews: {len(gmap_reviews)}')

# Print the number of unique users
print(f'Number of unique users: {num_users}')


# Out of 35,275 reviews, there are 34,921 unique users. This means that the vast majority of users have left only one review, making it less representative or meaningful to analyze their reviewing patterns (such as whether they tend to give high or low ratings). It is challenging to extract significant patterns or trends from this data. In other words, we believe it is difficult to conduct user-based analysis using the sample data.

# However, we believe that discovering that "user-based analysis is difficult" is itself an important finding. Let's break down this disappointing discovery step by step:
# 
# 1. In our sample data, we collected a total of 35,275 reviews, which were provided by 34,921 unique users. On average, each user has posted only 1.01 reviews, indicating a high level of dispersion and sparsity in user review behavior.
# 
# 2. Since most users have only provided one review, it is challenging to analyze behavior patterns at the user level. Even when setting a low "high-frequency user" threshold (e.g., 2 reviews), we can only identify 9 users. This falls short of our usual goal of identifying useful patterns from an active user base.
# 
# 3. This stands in stark contrast to platforms with strong social attributes, such as social media. The low frequency of interaction suggests that Google Maps has almost no social attributes. Users primarily use it as a tool to obtain information rather than as a social platform for frequent interaction.
# 
# 4. The lack of social attributes limits the effectiveness of Google Maps' recommendation system. Social media platforms' recommendation systems often rely on users' social relationships and consumption behaviors. In contrast, Google Maps lacks these social dimensions of data, making its recommendation system less effective compared to social media.
# 
# Overall, this sample Google Maps review data reflects a lack of social attributes, which directly limits the effectiveness and monetization potential of its recommendation system.

# Let's create a plot of the relationship between Review Rating and Times Business Being Recommended to further validate our hypothesis.

# In[ ]:


# Create a boxplot to visualize the distribution of review text length across different ratings
(ggplot(gmap_reviews, mapping=aes(x='factor(review_rating)', y='times_recommended')) +
    geom_boxplot() +  # Add a boxplot layer to the plot
    labs(
        title='Relationship Between Review Rating & Times Business Being Recommended',  # Add a title to the plot
        x='Review Rating',  # Label for the x-axis
        y='Times Business Being Recommended'  # Label for the y-axis
    ) +
    coord_flip() +
    theme_minimal() +  # Apply a minimalistic theme to the plot
    theme(
        figure_size=(8, 5.2),  # Set the size of the figure
        panel_background=element_rect(fill='white'),  # Set the panel background color to white
        plot_background=element_rect(fill='white'),  # Set the overall plot background color to white
        plot_title=element_text(hjust=0),  # Align the plot title to the left
        panel_grid_minor=element_line(color='lightgrey', size=0.4),  # Style the minor grid lines
    )
)


# The above plot shows that as the number of times a business is recommended increases, the review ratings given by users do not significantly improve.

# Remember that we approximated Review Count as customer foot traffic earlier? Let's create a plot to explore the relationship between Review Count and Times Business Being Recommended.

# In[ ]:


# Create a boxplot to visualize the distribution of review text length across different ratings
(ggplot(gmap_info, mapping=aes(x='times_recommended', y='review_count')) +
    geom_point() +  # Add a boxplot layer to the plot
    geom_smooth(color='red') +
    labs(
        title='Relationship Between Review Count & Times Business Being Recommended',  # Add a title to the plot
        x='Times Business Being Recommended',  # Label for the x-axis
        y='Review Count'  # Label for the y-axis
    ) +
    theme_minimal() +  # Apply a minimalistic theme to the plot
    theme(
        figure_size=(8, 5.2),  # Set the size of the figure
        panel_background=element_rect(fill='white'),  # Set the panel background color to white
        plot_background=element_rect(fill='white'),  # Set the overall plot background color to white
        plot_title=element_text(hjust=0),  # Align the plot title to the left
        panel_grid_minor=element_line(color='lightgrey', size=0.4),  # Style the minor grid lines
    )
)


# The above plot reveals a significant phenomenon: although the number of recommendations for certain businesses has increased, their review count has not correspondingly risen.

# These charts further support our perspective: in its current form, Google Maps' recommendation system has not successfully converted the number of recommendations into actual customer traffic or improved customer ratings for businesses. The effectiveness of Google Maps' recommendation system is limited by the lack of strong social attributes. Compared to recommendation systems that heavily rely on social interactions, Google Maps' recommendation feature may fall short in driving user behavior.
# 
# If Google Maps aims to enhance the effectiveness of its business model through its recommendation system, it might need to increase social interaction features to improve the real impact of recommendations.

# # Step 4: Summary of Meaningful Insights

# This step 4 contains the relatively meaningful insights found in this task. This step also serves as the presentation content of our task 4 video presentation. We put it here for the teaching team to refer to.

# ## Introduction

# Group 020 Members:
# - Ruiwen Chen
# - Zihan Yin

# Context of Analysis
# 
# This assignment involves analysis of datasets derived from Google Maps, focusing on both business and user perspectives. The goal is to uncover meaningful insights by exploring review patterns, metadata, and the potential impact of Google Maps' business model on its recommendation system. 

# Datasets Used:
# 1. Business Information From Main Google Review Data
# 2. Review Information From Main Google Review Data
# 2. MetaData
# 3. Combination of them

# ## Methodology

# - For the Pre-Processing, we perform data integration & split, converting data type and some feature engineering.
# - For the Analysis, we divide the analysis into two parts: first, univariate analysis on the datasets, and then multivariate analysis. Multivariate analysis is divided into two perspectives: business and user. We frequently use visualization throughout the process to help better understand the findings.

# ## Insights

# Here're 5 relatively meaningful insights we find.

# **Insight 1**:

# We observe significant differences in the number of reviews given at different hours of a day. 
# 

# In[ ]:


# Create a bar plot to visualize the frequency of reviews throughout the day
(ggplot(gmap_reviews_for_visual, mapping=aes(x='hour')) +
    geom_bar(fill='dodgerblue') +  # Create a bar plot with bars filled in 'dodgerblue' color
    labs(
        title='Frequency of Reviews Throughout the Day',  # Add a title to the plot
        x='Hour of the Day',  # Label for the x-axis
        y='Review Count'  # Label for the y-axis
    ) +
    theme_minimal() +  # Apply a minimalistic theme to the plot
    theme(
        figure_size=(8, 5.2),  # Set the size of the figure
        panel_background=element_rect(fill='white'),  # Set the panel background color to white
        plot_background=element_rect(fill='white'),  # Set the overall plot background color to white
        plot_title=element_text(hjust=0),  # Align the plot title to the left
        panel_grid_minor=element_line(color='lightgrey', size=0.4)  # Style the minor grid lines
    )
)


# This bar chart shows review counts peak during the midnight to early morning hours, decrease significantly in the morning, and rise again in the afternoon and evening. Overall, users tend to leave reviews outside of work hours, which aligns with typical daily routines.

# **Insight 2**:

# We find businesses tend to choose longer operating hours if they want to operate more days per week, thereby increasing customer accessibility.

# In[ ]:


# Create copies of gmap_info and metadata datasets
gmap_info_temp = gmap_info.copy()
metadata_temp = metadata.copy()

# Add a 'source' column to each dataset to identify the source of data
gmap_info_temp['source'] = 'gmap_info'
metadata_temp['source'] = 'metadata'

# Concatenate the two datasets, selecting only 'operating_days', 'average_daily_hours', and 'source' columns
combined_data = pd.concat([gmap_info_temp[['operating_days', 'average_daily_hours', 'source']],
                           metadata_temp[['operating_days', 'average_daily_hours', 'source']]])

# Create a boxplot to show the relationship between the number of operating days per week and average daily operating hours
(ggplot(combined_data, aes(x='factor(operating_days)', y='average_daily_hours', colour='source')) +
    geom_boxplot() +
    labs(
        title='Relationship Between Weekly Operating Days & Daily Operating Hour',  # Set the plot title
        x='Number of Weekly Operating Days',  # Label for the x-axis
        y='Average Daily Operating Hour'  # Label for the y-axis
    ) +
    facet_wrap('~source', ncol=1) +  # Use facet_wrap to display separate plots for each 'source'
    theme_minimal() +
    theme(
        figure_size=(9, 5),  # Set the figure size
        panel_background=element_rect(fill='white'),  # Set the panel background color to white
        plot_background=element_rect(fill='white'),  # Set the plot background color to white
        plot_title=element_text(hjust=0),  # Align the plot title to the left
        panel_grid_minor=element_line(color='lightgrey', size=0.4),  # Set minor grid lines
        strip_text=element_blank()  # Hide the subplot labels (facet strip texts)
    ) 
)


# We created separate plots for the relationship between the number of weekly days and the average daily hours using the main data & metadata.
# 
# For the red plot from main data, we find that as the number of weekly operating days increases, the average daily hours of businesses also increase. This trend is consistent across both datasets, which means that our sample data is representative.

# **Insight 2.1**:

# We use review count as a measuremeant of foot traffic, and find the operating model shown above cannot significantly boost foot traffic.

# In[ ]:


# Create a temporary grouping of review_count for visualization purposes
gmap_info_for_visual = gmap_info.copy()
gmap_info_for_visual['review_count_group'] = pd.qcut(gmap_info_for_visual['review_count'], q=4, labels=["Low", "Medium", "High", "Very High"])

# Create a boxplot to show the relationship between operating days, average daily hours, and review count group
(ggplot(gmap_info_for_visual, aes(x='factor(operating_days)', y='average_daily_hours', colour='review_count_group')) +
    geom_boxplot() +
    labs(
        title='Relationship Between Operating Hours, Days and Review Count Group',  # Set the plot title
        x='Average Daily Operating Hour',  # Label for the x-axis
        y='Number of Weekly Operating Days'  # Label for the y-axis
    ) +
    facet_wrap('~review_count_group') +  # Use facet_wrap to display separate plots for each review_count_group
    scale_color_brewer(palette='Set2', type='qual') +  # Use a color palette for better distinction between groups
    theme_minimal() +
    theme(
        figure_size=(10, 5),  # Set the figure size
        panel_background=element_rect(fill='white'),  # Set the panel background color to white
        plot_background=element_rect(fill='white'),  # Set the plot background color to white
        plot_title=element_text(hjust=0),  # Align the plot title to the left
        panel_grid_minor=element_line(color='lightgrey', size=0.4),  # Set minor grid lines
        strip_text=element_blank()  # Hide the subplot labels (facet strip texts)
    )
)


# The above figure is divided by review count, and we can see there's almost no difference in the trends of the 4 review count categories. This also shows customer traffic is more likely to be affected by other features.

# **Insight 2.2**:

# We find there're clear differences in customer foot traffic distribution in different cities & price levels.

# In[ ]:


# Aggregate the data to calculate the total number of reviews for each city
city_review_count = metadata.groupby('city')['num_of_reviews'].sum()

# Select the top 10 cities with the highest total review counts
top_cities = city_review_count.nlargest(10).index

# Filter the data to keep only the records from the top 10 cities
filtered_data = metadata[metadata['city'].isin(top_cities)]

# Apply a logarithmic transformation to the number of reviews to normalize the distribution
filtered_data['log_num_of_reviews'] = np.log1p(filtered_data['num_of_reviews'])

# Set the 'price_level' column as an ordered categorical variable, defining the order
price_level_order = ['Unknown Price Level', 'Low', 'Medium', 'High', 'Very High']
filtered_data['price_level'] = pd.Categorical(filtered_data['price_level'], categories=price_level_order, ordered=True)

# Create a boxplot to visualize the log-transformed review count distribution across the top 10 cities, faceted by price level
(ggplot(filtered_data, aes(x='city', y='log_num_of_reviews', color='price_level')) +
    geom_boxplot() +
    labs(
        title='Log-Transformed Review Count Distribution Across Top 10 Cities',  # Set the plot title
        x='City',  # Label for the x-axis
        y='Log of Review Count'  # Label for the y-axis
    ) +
    facet_wrap('~price_level') +  # Facet the plot by price level
    theme(
        figure_size=(10, 5),  # Set the figure size
        panel_background=element_rect(fill='white'),  # Set the panel background color to white
        plot_background=element_rect(fill='white'),  # Set the plot background color to white
        plot_title=element_text(hjust=0),  # Align the plot title to the left
        panel_grid_minor=element_line(color='lightgrey', size=0.4),  # Set minor grid lines
        axis_text_x=element_text(rotation=90, hjust=1),  # Rotate x-axis labels to prevent overlap
        legend_position="none"  # Hide the legend
    )
)


# This plot shows the distribution of review counts in the top 10 cities, with respective to different price levels. By applying a log transformation to the review counts, we can more clearly observe the impact of city and price level on customer foot traffic.

# **Insight 3**:

# During the user-based analysis, we find that, on average, each user has only 1.1 reviews.

# In[ ]:


# Calculate the number of unique users
num_users = gmap_reviews.index.get_level_values('user_id').nunique()

# Print the number of reviews (rows) in gmap_reviews
print(f'Number of reviews: {len(gmap_reviews)}')

# Print the number of unique users
print(f'Number of unique users: {num_users}')


# This indicates that Google Map lacks social attributes, making its recommendation system ineffective in boosting business customer traffic and ratings.

# In[ ]:


# Create a boxplot to visualize the distribution of review text length across different ratings
(ggplot(gmap_reviews, mapping=aes(x='factor(review_rating)', y='times_recommended')) +
    geom_boxplot() +  # Add a boxplot layer to the plot
    labs(
        title='Relationship Between Review Rating & Times Business Being Recommended',  # Add a title to the plot
        x='Review Rating',  # Label for the x-axis
        y='Times Business Being Recommended'  # Label for the y-axis
    ) +
    coord_flip() +
    theme_minimal() +  # Apply a minimalistic theme to the plot
    theme(
        figure_size=(8, 5),  # Set the size of the figure
        panel_background=element_rect(fill='white'),  # Set the panel background color to white
        plot_background=element_rect(fill='white'),  # Set the overall plot background color to white
        plot_title=element_text(hjust=0),  # Align the plot title to the left
        panel_grid_minor=element_line(color='lightgrey', size=0.4),  # Style the minor grid lines
    )
)


# Chart 1 shows even though the number of times a business is recommended increases, it does not lead to a significant improvement in user ratings.

# In[ ]:


# Create a boxplot to visualize the distribution of review text length across different ratings
(ggplot(gmap_info, mapping=aes(x='times_recommended', y='review_count')) +
    geom_point() +  # Add a boxplot layer to the plot
    geom_smooth(color='red') +
    labs(
        title='Relationship Between Review Count & Times Business Being Recommended',  # Add a title to the plot
        x='Times Business Being Recommended',  # Label for the x-axis
        y='Review Count'  # Label for the y-axis
    ) +
    theme_minimal() +  # Apply a minimalistic theme to the plot
    theme(
        figure_size=(8, 5),  # Set the size of the figure
        panel_background=element_rect(fill='white'),  # Set the panel background color to white
        plot_background=element_rect(fill='white'),  # Set the overall plot background color to white
        plot_title=element_text(hjust=0),  # Align the plot title to the left
        panel_grid_minor=element_line(color='lightgrey', size=0.4),  # Style the minor grid lines
    )
)


# Chart 2 shows more recommendations do not result in increased customer traffic.

# ## Conclusion

# The analysis shows that user engagement on Google Map is closely to daily routines, with review activity peaking outside of working hours. 
# 
# While businesses with longer operating hours and more days tend to be more accessible, but this does not significantly boost foot traffic or improve ratings. 
# 
# Additionally, customer traffic varies by city and price level, highlighting the importance of these factors.
# 
# A key finding is the average user leaves only 1.1 reviews, indicating a lack of social interaction on the platform. This limits the effectiveness of Google Map recommendation system in driving customer traffic. Future research could explore additional factors affecting business success and consider ways to enhance the platform's social features.
