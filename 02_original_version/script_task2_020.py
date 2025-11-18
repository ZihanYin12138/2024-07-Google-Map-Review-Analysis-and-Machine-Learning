#!/usr/bin/env python
# coding: utf-8

# <div class="alert alert-block alert-danger">
# 
# # FIT5196 Task 2 in Assessment 1
#     
# #### Student Name: Ruiwen Chen
# #### Student ID: 31512941
# 
# Date: 24/08/2024
# 
# Environment: JupyterNotebook
# 
# Libraries used:
# * os (for interacting with the operating system, included in Python xxxx) 
# * pandas 1.1.0 (for dataframe, installed and imported) 
# * multiprocessing (for performing processes on multi cores, included in Python 3.6.9 package) 
# * itertools (for performing operations on iterables)
# * nltk 3.5 (Natural Language Toolkit, installed and imported)
# * nltk.tokenize (for tokenization, installed and imported)
# * nltk.stem (for stemming the tokens, installed and imported)
# 
#     </div>

# <div class="alert alert-block alert-info">
#     
# ## Table of Contents
# 
# </div>
# 
# [1. Introduction](#Intro) <br>
# [2. Importing Libraries](#libs) <br>
# [3. Examining Input File](#examine) <br>
# [4. Unigram and Bigram](#load) <br>
# $\;\;\;\;$[4.1. Loading and Parsing Files](#tokenize) <br>
# $\;\;\;\;$[4.2. Tokenization](#whetev) <br>
# $\;\;\;\;$[4.3. Stemmer, Stopwords Removal to generate unigram and bigrams](#whetev) <br>
# $\;\;\;\;$[4.4. Genegrate numerical representation](#whetev1) <br>
# [5. Writing Output Files](#write) <br>
# $\;\;\;\;$[5.1. Vocabulary List](#write-vocab) <br>
# $\;\;\;\;$[5.2. Sparse Matrix](#write-sparseMat) <br>
# [6. Summary](#summary) <br>
# [7. References](#Ref) <br>

# <div class="alert alert-block alert-success">
#     
# ## 1.  Introduction  <a class="anchor" name="Intro"></a>

# This assessment concerns textual data, and the aim is to extract, process, and transform it into a proper format. The dataset provided is in the format of CSV and JSON files containing reviews and associated metadata for various businesses. The CSV file includes structured information, such as review counts and identifiers, while the JSON file provides detailed reviews, including user comments, timestamps, and other relevant information.
# 
# The task requires extracting review texts, processing them to ensure consistency and cleanliness (e.g., converting to lowercase, removing emojis, and filtering out non-English text), and then generating a vocabulary of unigrams and bigrams. These vocabularies will be used to create a sparse numerical representation of the data, which captures the frequency of terms within the reviews. This process involves several steps, including tokenization, stopword removal, stemming, and handling context-dependent stopwords.
# 
# The final output will include a vocabulary list sorted alphabetically and a sparse representation file that details the occurrence of terms in each business's review set. The goal is to prepare the data for further analysis, by ensuring it is in a structured and consistent format. 

# <div class="alert alert-block alert-success">
#     
# ## 2.  Importing Libraries  <a class="anchor" name="libs"></a>

# In this assessment, any python packages is permitted to be used. The following packages were used to accomplish the related tasks:
# 
# * **os:** to interact with the operating system, e.g. navigate through folders to read files
# * **re:** to define and use regular expressions
# * **pandas:** to work with dataframes
# * **multiprocessing:** to perform processes on multi cores for fast performance 
# * **itertools.chain:** Used to flatten lists of lists, simplifying the process of aggregating data from multiple sources.
# * **nltk:** The Natural Language Toolkit (nltk) is a comprehensive library for text processing. The specific modules used include:
# * **nltk.probability:** For calculating the frequency distribution of words.
#    * nltk.tokenize.RegexpTokenizer: For tokenizing text using regular expressions.
#    * nltk.tokenize.MWETokenizer: For tokenizing multi-word expressions (e.g., bigrams).
#    * nltk.stem.PorterStemmer: For stemming words to their root forms, which helps in reducing the vocabulary size by grouping similar words.
#    * nltk.util.ngrams: For generating n-grams, which are sequences of n tokens (words) used for bigram generation.
#    * nltk.collocations.BigramCollocationFinder: For finding bigrams (pairs of words that frequently occur together) and calculating their association measures.
#    * nltk.collocations.BigramAssocMeasures: For measuring the strength of association between words in bigrams.
# * **collections.defaultdict:** A dictionary subclass that provides a default value for a nonexistent key, which simplifies counting word frequencies and handling missing data.
# * **sklearn.feature_extraction.text.CountVectorizer:** A crucial component of the scikit-learn library used for converting a collection of text documents into a matrix of token counts. This is particularly useful in natural language processing tasks where numerical representations of text (such as word counts or term frequencies) are required for further analysis or machine learning.

# In[ ]:


#!pip uninstall nltk -y
#!pip install nltk==3.8.1
import os
import re
import langid
import json
import pandas as pd
import multiprocessing
from itertools import chain
import nltk
from nltk.probability import *
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import MWETokenizer
from nltk.stem import PorterStemmer
from nltk.util import ngrams
from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer


# -------------------------------------

# <div class="alert alert-block alert-success">
#     
# ## 3.  Examining Input File <a class="anchor" name="examine"></a>

# In[ ]:


task1_csv='task1_020.csv'
task1_json='task1_020.json'


# Let's examine what is the content of the file. For this purpose, we would like to check what information is stored in the csv and json files. 
# * As the files are too large to open them all in JupyterNotebook, we firstly print the head 5 rows and columns of the csv to see what information it contains. By knowing that it helps manipulate and filter the data for further processing.
# * To inspect json structure, we break it down as JSON file likely contains a more complex data structure, such as nested dictionaries and lists, related to reviews and other metadata. Understanding the structure of this value is essential for further analysis, such as extracting review texts.

# In[ ]:


# Load the CSV file
data_csv = pd.read_csv('task1_020.csv')
print(data_csv.head(5))
print(len(data_csv))


# In[ ]:


# Load the JSON file
with open('task1_020.json', 'r') as file:
    data_json = json.load(file)

# Inspect the JSON structure
print(list(data_json.keys())[:1])  # Display the first 1 keys
print({key: data_json[key] for key in list(data_json.keys())[:5]})  # Display the first 5 key-value pairs


# Having parsed the CSV and JSON file, the following observations can be made
# * The CSV file contain structured data, includes "gmap_id", "review_count", "review_text_count" and "resonse_count".
# * The JSON file will be a dictionary where each key corresponds to a unique gmap_id.For each gmap_id, the dictionary will contain
#   * "reviews": A list of dictionaries, where each dictionary represents a single review.
#   * "earliest_review_date": A string representing the date and time of the earliest review for this gmap_id.
#   * "latest_review_date": A string representing the date and time of the latest review for this gmap_id.
#     * Each review in the "reviews" list is a dictionary with the following keys:
#       * "user_id"
#       * "time"
#       * "review_rating"
#       * "review_text"
#       * "if_pic"
#       * "pic_dim"
#       * "if_response"
# 

# <div class="alert alert-block alert-success">
#     
# ## 4.  Unigram and Bigram <a class="anchor" name="load"></a>

# <div class="alert alert-block alert-warning">
#     
# ### 4.1. Loading and Parsing File <a class="anchor" name="tokenize"></a>

# It is noteiced that file contains contains many columns, we want gamp_id and review text only. from all the gmap_id, we only want the gmap_id that has at least 70 reveiw texts. 
# 
# In this section, we will load and parse the data, filtering it according to specific criteria.
# * **Criteria 1**: Filter the data to include only those entries with review_text_count greater than or equal to 70.
# * **Criteria 2**: Filtering for non-empty review texts

# To effectively store the extracted review data, we begin by initializing an empty dictionary called extracted_data. This dictionary is designed to hold the processed review information for each gmap_id as its keys. We then iterate over each gmap_id in the data_json dictionary, which contains all the original review data. 
# 
# As we loop through each review in the reviews list, we apply two key conditions: first, we check if the gmap_id is present in the gmap_id_list, a pre-filtered list that includes only those gmap_ids that meet specific criteria, such as having at least 70 reviews. Second, we ensure that the review_text field is not empty.
# 
# For reviews that satisfy both conditions, the relevant information is extracted and processed. Specifically, the review_text is converted to lowercase to maintain uniformity and prepare it for future tokenization. Additionally, the original time field is retained to allow for chronological analysis of the reviews.
# 
# Once all reviews for a given gmap_id have been processed, the extracted_reviews list is stored in the extracted_data dictionary under its corresponding gmap_id key. This process is repeated for each gmap_id in the original JSON data. This structured approach filter out unused data and extracts specific, relevant information. This targeted extraction process enhances the quality of the data, making it more manageable and suitable for future analysis.

# In[ ]:


filter_text=data_csv[data_csv['review_text_count'] >= 70] # Filter review_text that is greater than or equal to 70
gmap_id_list=filter_text['gmap_id'].tolist() #get those filter_text's gamp_id


# Initialize an empty dictionary to store the extracted data
extracted_data = {}

# Iterate over each gmap_id in the original JSON data
for gmap_id, data in data_json.items():
    reviews = data['reviews']  # Get the list of reviews for the current gmap_id
    
    # Iterate over each review and extract the desired information
    extracted_reviews = []
    # Iterate over each review and extract the desired information
    for review in reviews:
        # Check if the gmap_id is in the filtered gmap_id_list and the review text is not "None"
        if gmap_id in gmap_id_list and review["review_text"] !="None":
            extracted_reviews.append({
                "text_reviews": review["review_text"].lower(),# Convert review text to lowercase for future tokenizer
                "time": review["time"] # Keep the original timestamp for tokenize them on the same day
            })
    
            # Store the extracted information under the corresponding gmap_id
            extracted_data[gmap_id] = extracted_reviews

print("\n extracted_data (sample):")
for gmap_id, vocab in list(extracted_data.items())[:1]:  # Limit to first 5 entries
        print(f"{gmap_id}: {vocab[:10]}")  # Print first 10 tokens for each gmap_id


# Let's examine the dictionary generated. For counting the total number of reviews extracted, check whether the number of review_texts are all greater than or equal to 70.
# 

# In[ ]:


if all(len(value) >=70 for value in extracted_data.values()):
    print("All the businesses have at least 70 text reviews.")
else:
    print("There exists at least one business with fewer than 70 text reviews.")


# <div class="alert alert-block alert-warning">
#     
# ### 4.2. Sort Text by Date <a class="anchor" name="tokenize"></a>

# In this section, we want a dictionary that has gmap_id as it's key, and its' value is a single string for all reviews of the day concatenated to each other. The result is a dictionary where all reviews from the same day are concatenated, making it easier to analyze or process the data further.

# To further process the extracted data, we begin by initializing an empty dictionary called concatenated_dict, which will eventually hold the concatenated review texts organized by gmap_id and their respective dates. The main reason for creating `concatenated_dict` is to organize the review texts by date. By concatenating all reviews from the same day, it is expected to reduce the complexity of later processing steps, such as tokenization, bigram extraction, or sentiment analysis. By initialising `concatenated_dict`, it simplifies the structure of your data, allowing for easier and more meaningful analysis of how the content of reviews changes daily.
# 
# For each gmap_id, we create another empty dictionary named date_text_dict to store review texts associated with specific dates. This dictionary will have dates as its keys and concatenated review texts as its values.
# 
# Within the outer loop, an inner loop iterates through each review in the list of reviews corresponding to the current gmap_id. For each review, the time field is split to extract the date in YYYY-MM-DD format. The review text is also accessed and stored in the variable review_text.
# 
# To handle multiple reviews on the same day, we use an if statement to check whether the extracted review_date already exists as a key in the date_text_dict. If it does, the current review text is appended to the existing text, separated by a space, effectively concatenating all reviews from the same date into a single string. If the review_date does not exist in the dictionary, a new key-value pair is created, with the date as the key and the review text as the value.

# In[ ]:


# line initializes an empty dictionary
time_sorted_dict = {}

# loop iterates over each gmap_id and its associated reviews from the extracted_data dictionary
for gmap_id, reviews in extracted_data.items():
    # Initialize an empty dictionary to store results for the current gmap_id
    date_text_dict = {}
    
    # inner loop iterates through each review in the list of reviews for the current gmap_id.
    for review in reviews:
        # Extract the date (YYYY-MM-DD)
        review_date = review["time"].split(" ")[0]
        review_text = review["text_reviews"]
        
        # Use an if statement to check if review_date already exists in the dictionary
        if review_date in date_text_dict:
            # If the date already exists, append the new review text to the existing text
            date_text_dict[review_date] += " " + review_text
        else:
            # If the date doesn't exist, create a new key-value pair
            date_text_dict[review_date] = review_text
    
    # Store the processed results in the final result dictionary
    time_sorted_dict[gmap_id] = date_text_dict
    
# Print few output
for date, text in list(time_sorted_dict[list(time_sorted_dict.keys())[0]].items())[:20]:  
    print(f"  Date: {date}, Review Text: {text[:100]}...")  # Print only the first 100 characters of the review text


# <div class="alert alert-block alert-warning">
#     
# ### 4.3. Tokenization <a class="anchor" name="tokenize"></a>

# The above operation results in a dictionary with gmap_id representing keys and a single string for all reviews of the day concatenated to each other. 

# Our goal now is to transform concatenated review texts into structured tokens, specifically unigrams, which will be crucial for subsequent analysis. To achieve this, we utilize the `RegexpTokenizer` from the `nltk` library, initializing it with a regular expression (`r"[a-zA-Z]+"`) to ensure that only alphabetic characters are retained. This approach allows us to filter out numbers, punctuation, and special characters, enabling us to concentrate on the meaningful content of the reviews without being distracted by non-informative symbols.
# 
# We iterating over each `gmap_id` in the `concatenated_dict`, accessing the corresponding dictionary that contains review texts organized by date. For each date under a specific `gmap_id`, we retrieve the concatenated review text and tokenize it into individual words using the `tokenize()` method of `RegexpTokenizer`. The resulting list of words represents the meaningful content of all reviews written on that specific date. After tokenization, we store the list of tokens in the `tokenized_reviews` dictionary under the corresponding date, ensuring that each date's review content is processed independently.
# 
# Finally, we store the `tokenized_reviews` dictionary, which contains date-based tokenized reviews, in the `tokenizer_dict` under the appropriate `gmap_id`. This hierarchical structure—`gmap_id` -> date -> tokens—allows us to maintain the contextual relationships in the data, making it easy to access and analyze the tokenized content at different levels of granularity. 

# In[ ]:


tokenizer_dict={}

# Iterate over each gmap_id and its corresponding date-based review text dictionary in concatenated_dict
for gmap_id, date in time_sorted_dict.items():
    tokenized_reviews = {} # Initialize an empty dictionary to store tokenized reviews for each date
    
    # Iterate over each date and its corresponding review text in date_reviews
    for date, text in date.items():
        tokenizer = RegexpTokenizer(r"[a-zA-Z]+") # Initialize the tokenizer to only keep alphabetic words
        tokens = tokenizer.tokenize(text) # Tokenize the text into individual words
        tokenized_reviews[date]=tokens # Store the tokenized words under the corresponding date
        
    # Store the tokenized reviews under the corresponding gmap_id
    tokenizer_dict[gmap_id]=tokenized_reviews
    
#print output for checking    
for date, tokens in list(tokenizer_dict[list(tokenizer_dict.keys())[0]].items())[:10]: 
    print(date, tokens[:100])  


# At this stage, all reviews for each gmap_id are tokenized and are stored as a value in the new dictionary (separetely for each day).
# 
# -------------------------------------

# <div class="alert alert-block alert-warning">
#     
# ### 4.4. Stemmer, Stopwords Removal to generate unigram and bigrams <a class="anchor" name="whetev"></a>

# In this section, We focus on refining our vocabulary by filtering out unwanted words based on specific criteria. These criterias are: 
# * **Stopwords**
# * context-independent: context-independent stop words list(stopwords_en.txt) will be used
# * context-dependent: words that appear in more than 95% of the businesses that have at least 70 text reviews.
# * rare tokens: words that appear in less than 5% of the businesses that have at least 70 text reviews.
# * Length: tokens with a length less than 3 should be removed from the vocab
# * **Stemmer**
# * PorterStemmer will be used
# * **Bigrams**
# * First 200 meaningful bigrams will be included in the vocab using PMI measure

# For bigrams, we choose to stem only unigrams and not bigrams. This decision is grounded in the necessity to preserve the semantic integrity and contextual meaning of the bigrams. We find that stemming unigrams helps in reducing the dimensionality of the vocabulary by merging words with similar roots, which is beneficial for simplifying and generalising text data. However, applying the same process to bigrams would strip away essential context and lead to a significant loss in meaning, rendering the bigrams less informative and harder to interpret.
# For instance, the bigram "data mining" potentially becoming "data mine" after stem, which would not only disrupt the readability but also introduce ambiguity, as the stemmed version might not accurately represent the original concept.

# In[ ]:


# Load context-independent stopwords from a file into a set
with open('stopwords_en.txt', 'r') as file:
    stopwords_independent = set(file.read().splitlines())
    
stemmer = PorterStemmer()  # Initialize the PorterStemmer for stemming words
word_frequency = defaultdict(int)  # Initialize a dictionary to count word frequencies
total_id = len(tokenizer_dict)  # Get the total number of gmap_id entries
final_gmap_dict = {}  # Initialize a dictionary to store gmap_id and filtered_vocabulary

# Iterate through the tokenizer_dict, which contains tokenized reviews
for gmap_id, reviews in tokenizer_dict.items():
    stemmed_tokens = []  # Use a list to store the final filtered vocabulary for this gmap_id
    no_stemmed_tokens= []  # Use a list to store all tokens that not stemmed, for bigrams
    
    for date, tokens in reviews.items():
        # Stem each token that is not in the context-independent stopwords list
        stem = [stemmer.stem(token) for token in tokens if token.lower() not in stopwords_independent]
        # a list to store all tokens that haven't been stemmed, for bigrams
        filtered_tokens = [token for token in tokens if token.lower() not in stopwords_independent]
        
        # Add all stemmed tokens to filtered_vocabulary list (keeping repetitions)
        stemmed_tokens.extend(stem)
        # Add all tokens to all_tokens list for bigram processing
        no_stemmed_tokens.extend(filtered_tokens)
        
    # Calculate word frequency based on occurrences in filtered_vocabulary list
    for word in stemmed_tokens:
        word_frequency[word] += 1

    # Identify context-dependent stopwords (appear in more than 95% of gmap_ids)
    stopwords_dependent = {word for word, count in word_frequency.items() if count / total_id > 0.95}

    # Determine rare words that appear in less than 5% of the gmap_ids
    rare_threshold = 0.05 * total_id
    rare_words = {word for word, count in word_frequency.items() if count < rare_threshold}

    # Identify short words (length less than 3 characters)
    short_words = {word for word, count in word_frequency.items() if len(word) < 3}

    # Filter out the stopwords (independent and dependent), rare words, and short words
    unigram_vocab = [
        word for word in stemmed_tokens 
        if word not in stopwords_independent 
        and word not in stopwords_dependent 
        and word not in rare_words 
        and word not in short_words
    ]
    # Filter out the stopwords (independent and dependent), rare words, and short words
    bigram_list_filtered = [
        word for word in no_stemmed_tokens
        if word not in stopwords_independent 
        and word not in stopwords_dependent 
        and word not in rare_words 
        and word not in short_words
    ]
 
        # Initialize BigramAssocMeasures 
    bigram_measures = BigramAssocMeasures()

        # Create a BigramCollocationFinder from the list of all tokens
    finder = BigramCollocationFinder.from_words(bigram_list_filtered)

        # Use the PMI measure to find the top 200 bigrams with the highest PMI scores
    bigrams_with_pmi = finder.nbest(bigram_measures.pmi, 200)

        # Join the words in each bigram with an underscore to form a single token and store these in a set
    bigram_vocab = ['_'.join(bigram) for bigram in bigrams_with_pmi]
    
    #Check whether the bigrams are in the text, if in the text, keep it, if not, remove
    token_bigrams = list(ngrams(no_stemmed_tokens, 2))
    
    valid_bigrams = []
    for bigram in bigram_vocab:
        word1, word2 = bigram.split('_')
        if (word1, word2) in token_bigrams:
            valid_bigrams.append(bigram)
            
    final_gmap_dict[gmap_id] = unigram_vocab + valid_bigrams
            
    print(f"The len of the valid bigram are {len(valid_bigrams)}\n")
    print(f"The len of the unigram are {len(unigram_vocab)}\n")
    
print(f"The len of the total valid bigram are {len(valid_bigrams)}\n")
print(f"The len of the total unigram are {len(unigram_vocab)}\n")


# To check the frequency of each bigram, we create a loop that iterates through the bigrams_with_pmi, converts them to strings, and counts their occurrences in the corpus.

# <div class="alert alert-block alert-warning">
#     
# ### 4.4. Generate numerical representation<a class="anchor" name="bigrams"></a>

# One of the tasks is to generate the numerical representation for all tokens in abstract.  In this section, we will use CountVectorizer to generate the sparse numerical representation. 
# 
# First, we prepare the data by joining the tokens into a single string for each `gmap_id`. This process is necessary because `CountVectorizer` expects the input data to be in the form of text documents, where each document is represented as a string. The `text_data` list is created by iterating over the tokenized reviews stored in `tokenizer_dict`, and for each entry, the tokens are joined together with spaces to form a single string.
# 
# By setting the analyzer parameter to 'word', we specify that we want to analyze individual words (or tokens) as features. 
# 
# After initializing the CountVectorizer, we fit the model to our textual data and simultaneously transform this data into a sparse matrix. This sparse matrix efficiently stores the frequency of each token across the entire dataset, with rows corresponding to different gmap_ids and columns representing the various tokens identified by the CountVectorizer.
# 
# To facilitate further processing, we retrieve the feature names using the get_feature_names_out() function. These feature names correspond to the tokens that have been identified and counted during the transformation process.
# 
# Finally, we convert the sparse matrix to a dense array using the toarray() function. This conversion allows us to easily iterate over the matrix and extract the frequency of each token for each gmap_id. By working with this dense representation, we can conveniently access and manipulate the data as needed for subsequent steps in our analysis.
# 
# 

# In[ ]:


# Prepare the data by joining tokens into a string for each gmap_id
text_data = [' '.join(tokens) for tokens in final_gmap_dict.values()] 

# Initialize CountVectorizer with the combined vocabulary
vectorizer = CountVectorizer(analyzer='word', vocabulary=set(unigram_vocab + valid_bigrams))

# Fit the model and transform the data into a sparse matrix
data_features = vectorizer.fit_transform(text_data)

# Get the feature names, which correspond to the tokens in our final vocabulary
feature_names = vectorizer.get_feature_names_out()

# Print a sample of the feature names for verification
print("The lens of feature_names:")
print(len(feature_names))


# At this stage, we have a dictionary of tokenized words, whose keys are indicative of gmap_id and values are tokenized words includes both bigrams and unigrams.
# 
# -------------------------------------

# <div class="alert alert-block alert-success">
#     
# ## 5. Writing Output Files <a class="anchor" name="write"></a>

# In this session files need to be generated:
# * Vocabulary list
# * Sparse matrix (count_vectors)
# 
# This is performed in the following sections.

# <div class="alert alert-block alert-warning">
#     
# ### 5.1. Vocabulary List <a class="anchor" name="write-vocab"></a>

# List of vocabulary should also be written to a file, sorted alphabetically, with their reference codes in front of them. This file also refers to the sparse matrix in the next file. For this purpose, we use sorted() function.

# In[ ]:


# Sort the feature names alphabetically and assign indices
sorted_vocab = sorted(feature_names)
vocab_dict = {word: index for index, word in enumerate(sorted_vocab)}

# Write the sorted vocabulary with reference codes to a file
with open('020_vocab.txt', 'w') as vocab_file:
    for word, index in vocab_dict.items():
        vocab_file.write(f"{word}:{index}\n")


# <div class="alert alert-block alert-warning">
#     
# ### 5.2. Sparse Matrix <a class="anchor" name="write-sparseMat"></a>

# For writing the sparse matrix representation of the text data for each `gmap_id` into the `020_countvec.txt` file, we begin by calculating the frequency of words for each document. We have already tokenized and processed our data, and now we need to represent this data numerically in a sparse format.
# 
# First, we iterate over each `gmap_id` in `final_gmap_dict` to get the corresponding row from the sparse matrix, `data_features`. This matrix was generated by fitting the `CountVectorizer` model on our tokenized text data. Each row in this matrix corresponds to a specific `gmap_id`, and the columns represent the frequencies of different words or bigrams in our vocabulary.
# 
# For each `gmap_id`, we convert its corresponding row in the sparse matrix to a dense array using the `.toarray()` method. This conversion allows us to easily iterate over the word frequencies associated with that particular `gmap_id`.
# 
# Next, we initialize an empty list `freq_list` to store the frequency pairs for each word or bigram that occurs in the document. We iterate over each index in the row, checking if the frequency at that index is greater than zero. If it is, we retrieve the word corresponding to that index using `feature_names`, which holds all the words and bigrams in our vocabulary. We then look up the correct index for this word from `vocab_dict`, which maps each word to its corresponding index as written in the `020_vocab.txt` file.
# 
# We append the index and its corresponding frequency as a pair (formatted as `index:frequency`) to `freq_list`. Once all the non-zero frequencies for that `gmap_id` have been processed, we join the `freq_list` into a single string, prefixed by the `gmap_id`.
# 
# Finally, this string is written to the `020_countvec.txt` file, where each line corresponds to a `gmap_id`, followed by the indices and frequencies of the words and bigrams that appeared in the document. This format ensures that each document is represented concisely, with only the non-zero frequencies being recorded, making the matrix sparse and efficient for storage and processing. 

# In[ ]:


with open('020_countvec.txt', 'w') as f:
    for i, gmap_id in enumerate(final_gmap_dict.keys()):
        # Get the corresponding row in the sparse matrix for the current gmap_id
        row = data_features[i].toarray()[0]
        
        # Generate index:frequency pairs, only for non-zero frequencies
        freq_list = []
        for index, freq in enumerate(row):
            if freq > 0:
                # Get the correct index from vocab_dict
                word = feature_names[index]
                correct_index = vocab_dict[word]
                
                # Append the correct index:frequency pair
                freq_list.append(f"{correct_index}:{freq}")
        
        # Join the frequency list into a single string and write to the file
        line = f"{gmap_id}, " + ", ".join(freq_list) + "\n"
        f.write(line)


# -------------------------------------

# <div class="alert alert-block alert-success">
#     
# ## 6. Summary <a class="anchor" name="summary"></a>

# To summarise Task 2, we have undertaken the following steps:
# 
# Firstly, we examined the input files, including CSV and JSON files, to understand their content and structures. This initial analysis allowed us to identify the key information required for further processing.
# 
# Next, we parsed the files and filtered the data to focus on the review texts from businesses that have at least 70 reviews. By doing so, we ensured that the dataset was refined and manageable, reducing complexity and focusing our analysis on the most relevant data.
# 
# We then proceeded to generate the unigram and bigram lists. This involved several preprocessing steps, including word tokenization using a specific regular expression, removing both context-independent and context-dependent stopwords, stemming tokens using the Porter stemmer, and filtering out rare and short tokens. The resulting vocabulary, containing both unigrams and bigrams, was sorted alphabetically and output as `vocab.txt`.
# 
# Finally, we generated the sparse numerical representation of the data using the `CountVectorizer()` function. This representation captured the frequency of each token, with the indices corresponding to those in the `vocab.txt` file. The output was saved in `countvec.txt`, formatted to link each `gmap_id` with its corresponding token frequencies.
# 
# By following these steps, we ensured that the data was systematically processed, resulting in a clean and structured representation ready for further analysis.
# 
# 
# 
# 

# -------------------------------------

# <div class="alert alert-block alert-success">
#     
# ## 7. References <a class="anchor" name="Ref"></a>

# [1] Pandas dataframe.drop_duplicates(), https://www.geeksforgeeks.org/python-pandas-dataframe-drop_duplicates/, Accessed 13/08/2022.
# 
# 

# ## --------------------------------------------------------------------------------------------------------------------------
