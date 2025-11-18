#!/usr/bin/env python
# coding: utf-8

# <div class="alert alert-block alert-success">
#     
# # FIT5196 Task 1 in Assessment 1
# #### Student Name: Zihan Yin
# #### Student ID: 34502297
# 
# Date: 2024.08.10
# 
# Environment: Python 3.12.4
# 
# Libraries used:
# * pandas
# * json
# * re
# * os
#     
# </div>

# <div class="alert alert-block alert-danger">
#     
# ## Table of Contents
# 
# </div>    
#     
# [1. Introduction](#1) <br>
# [2. Importing Libraries](#2) <br>
# [3. Raw Data Loading & Overview](#3) <br>
# $\;\;\;\;$[3.1 `.txt` Files Loading & Overview](##3.1) <br>
# $\;\;\;\;$[3.2 `.xlsx` File Loading & Overview](##3.2) <br>
# [4. Parsing Raw Data to a DataFrame](#4) <br>
# $\;\;\;\;$[4.1 Parsing `.txt` data](#latin) <br>
# $\;\;\;\;\;\;\;\;$[4.1.1 Cleaning Tags](#4.1.1) <br>
# $\;\;\;\;\;\;\;\;$[4.1.2 Replacing Tags](#replacing_tags) <br>
# $\;\;\;\;\;\;\;\;$[4.1.3 Extracting Records](#extracting_records) <br>
# $\;\;\;\;$[4.2 Parsing `.xlsx` data](#latin) <br>
# $\;\;\;\;$[4.3 Combining DataFrames Together](#latin) <br>
# [5. Data Processing](#write) <br>
# $\;\;\;\;$[5.1 Creating the Required Columns](#converting_data_type) <br>
# $\;\;\;\;$[5.2 Converting Data Type](#focus_text) <br>
# $\;\;\;\;$[5.3 Focus on `text` Column](#focus_text) <br>
# [6. Generate Output Files](#write) <br>
# $\;\;\;\;$[6.1 Writing to `.csv` File](#csv_file) <br>
# $\;\;\;\;$[6.2 Writing to `.json` File](#json_file) <br>
# $\;\;\;\;$[6.3 Verification of the Generated Files](#verification) <br>
# [7. Summary](#summary) <br>
# [8. References](#Ref) <br>

# <div class="alert alert-block alert-warning">
# 
# ## 1 Introduction  <a id='1'></a>
#     
# </div>

# This assignment mainly involves extracting records from messy data, converting them into a DataFrame, performing text preprocessing, and conducting exploratory data analysis.
# 
# This `.ipynb` file serves as Task 1 of this assignment. The main task is to convert the approx-semi-structured content in the given `.txt` and `.xlsx` files into a DataFrame suitable for subsequent analysis and output it in `.csv` and `.json` formats.

# <div class="alert alert-block alert-warning">
#     
# ## 2 Importing Libraries  <a id='2'></a>
#  </div>

# To complete the above tasks, we will use the following packages:
# * **pandas:** For various data manipulations, including but not limited to reading files & processing data.
# * **json:** For exporting output in `.json` format.
# * **re:** Mainly for extracting records from `.txt` files.
# * **os:** Used when reading multiple `.txt` files in batches.

# In[ ]:


import pandas as pd
import json
import re
import os


# <div class="alert alert-block alert-warning">
# 
# ## 3 Raw Data Loading & Overview <a id='3'></a>
# 
#  </div>

# Let's first read the raw data and see what they look like.

# <div class="alert alert-block alert-info">
#     
# ### 3.1 `.txt` Files Loading & Overview <a id='3.1'></a>

# Read all 15 `.txt` files and combine them into a single string. Then Display the first 1000 characters of the string.

# In[ ]:


# Path to the folder containing the`.txt files
folder_path = 'student_group020/'

# Retrieve the paths of all the .txt files and store in a list
file_list = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.txt')]

# Initialize an empty string to store the combined text
combined_text = ""

# Read each file and append its text to the combined text
for file in file_list:
    with open(file, 'r', encoding='utf-8') as f:
        combined_text = combined_text + f.read() + "\n"  # Read the file content and add a newline character

# Print the first 1000 characters of the combined text to avoid overly long output
print(combined_text[:2000])


# Although the content of the `.txt` files is presented in `.xml` file format, after observation, we found that there are many inconsistencies in the tag format. This means that it is difficult to read the data using the `pd.read_xml` function.
# 
# The following output shows the frequency of each tag in the string, with a total of **166 different tags**.

# In[ ]:


# Extract all tags using regex
# The regex pattern can find all the tags in `combined_text`
tags = re.findall(r'<[^<]*?>', combined_text)

# Count the frequency of each tag
tag_counts = pd.Series(tags).value_counts()

# Display the number of tags and the first 20 tags with highest frequency
print(f'The Number of Tags in the Raw Data: {len(tag_counts)}')
print(tag_counts.head(20))


# <div class="alert alert-block alert-info">
#     
# ### 3.2 `.xlsx` File Loading & Overview <a id='3.2'></a>

# Read the 16 sheets from `group020.xlsx`, and use `'Sheet0'` as an example to display.
# 
# We found that there are entire rows and columns with null values in the raw data.

# In[ ]:


# Read all sheets from the .xlsx file into a dictionary of DataFrames
excel_sheets = pd.read_excel("student_group020/group020.xlsx", sheet_name=None)

# Print the names of all the sheets in the .xlsx file
print(list(excel_sheets.keys()))

# Display the first 5 rows of the sheet "Sheet0"
excel_sheets["Sheet0"].head(5)


# -------------------------------------

# <div class="alert alert-block alert-warning"> 
# 
# ## 4 Parsing Raw Data to DataFrame <a id='4'></a>
# 
# </div>

# After having a general understanding of the raw data, let's convert them into structured data (DataFrame) that can be processed.

# <div class="alert alert-block alert-info">
#     
# ### 4.1 Parsing `.txt` data <a class="anchor" name="Reg_Exp"></a>

# After observing the `.txt` files in Section 3.1, we established the following parsing process:
# 
# 0. Extract all tags from the combined text and store them in a list (completed in Section 3.1).
# 1. For each tag in the list:
#     * Convert to lowercase
#     * Remove all spaces
#     * Remove all punctuation within '<>' except for the first '/'
#     * Merge ambiguous tags into a unified tag
# 2. Use a mapping dictionary to replace the original tags in the combined text with the cleaned tags.
# 3. Use regex to extract each record and store it in a list, then convert this list into a `pd.DataFrame`.

# <div class="alert alert-block alert-success">
# 
# #### 4.1.1 Cleaning Tags <a id='4.1.1'></a>

# To batch clean each tag in the list, define a helper function `clean_tag`:

# In[ ]:


# Define a helper function to clean each tag
def clean_tag(tag):
    # 1. Convert tag to lowercase
    tag = tag.lower()
    
    # 2. Remove all spaces & all punctuation inside the tag except for the first '/' on the left
    #    The 1st re.sub() removes all characters that are not letters, numbers, or the characters <>, and /
    #    The 2nd re.sub() replaces multiple '/' with a single slash
    tag = re.sub(r'[^<>/a-zA-Z0-9]+', r'', tag)
    tag = re.sub(r'/{2,}', r'/', tag)

    return tag

# Process all tags using the helper function clean_tag
cleaned_tags = [clean_tag(tag) for tag in tags]
cleaned_tags  # Display the cleaned tags


# Tags have been converted to a unified format. However, after examining the unique values, we found that some tags are ambiguous. 
# 
# For example, `<time>` and `<date>` actually refer to the same tag.

# In[ ]:


print(set(cleaned_tags))


# To deal with the situation presented above, we created a mapping dictionary to merge these ambiguous tags.

# In[ ]:


# This dictionary is used to normalize various forms of tags that represent the same concept
# by mapping them to a consistent, unified tag.

tag_mapping = {
    # map to <gmap_id> & </gmap_id>
    '<gmapid>': '<gmap_id>',
    '</gmapid>': '</gmap_id>',
    
    # map to <user_id> & </user_id>
    '<userid>': '<user_id>',
    '</userid>': '</user_id>',
    '<user>': '<user_id>',
    '</user>': '</user_id>',
    
    # map to <name> & </name>
    '<username>': '<name>',
    '</username>': '</name>',
    '<name>': '<name>',
    '</name>': '</name>',
    
    # map to <time> & </time>
    '<date>': '<time>',
    '</date>': '</time>',
    '<time>': '<time>',
    '</time>': '</time>',
    
    # map to <rating> & </rating>
    '<rate>': '<rating>',
    '</rate>': '</rating>',
    '<rating>': '<rating>',
    '</rating>': '</rating>',
    
    # map to <text> & </text>
    '<review>': '<text>',
    '</review>': '</text>',
    '<text>': '<text>',
    '</text>': '</text>',
    
    # map to <pics> & </pics>
    '<pictures>': '<pics>',
    '</pictures>': '</pics>',
    '<pics>': '<pics>',
    '</pics>': '</pics>',
    
    # map to <resp> & </resp>
    '<response>': '<resp>',
    '</response>': '</resp>',
    '<resp>': '<resp>',
    '</resp>': '</resp>',
}

# Iterate over the cleaned tags and replace each tag with its mapped tag
# If a tag does not have a mapping, it remains unchanged
cleaned_tags = [tag_mapping.get(tag, tag) for tag in cleaned_tags]
cleaned_tags  # Display the cleaned tags


# Show the frequency of tags after cleaning.

# In[ ]:


# Count the frequency of each tag
tag_counts = pd.Series(cleaned_tags).value_counts()

# Display the tags and the corresponding frequency
print(f'The Number of Tags After Cleaning: {len(tag_counts)}')
print(tag_counts)


# <div class="alert alert-block alert-success">
# 
# #### 4.1.2 Replacing Tags

# Use a mapping dictionary (another one) to replace the original tags in the combined text with the cleaned tags.

# In[ ]:


# Create a dictionary that maps each original tag to each cleaned tag
tag_mapping_dict = dict(zip(tags, cleaned_tags))

# Replace the original tags in the text based on the tag_mapping_dict
# Iterate over the mapping dictionary and replace each original tag with the corresponding cleaned tag
for original_tag, cleaned_tag in tag_mapping_dict.items():
    combined_text = combined_text.replace(original_tag, cleaned_tag)

# Print the first 1000 characters of the combined_text after replacing
print(combined_text[:2000])

# Remove all `\n` from the combined_text
combined_text = combined_text.replace('\n', '')


# <div class="alert alert-block alert-success">
# 
# #### 4.1.3 Extracting Records

# Use regex to extract each record.
# 
# The regex pattern `'<tag>(.*?)</tag>'` can find the content between each pair of tags. We found that the content between many tags is the string 'None', which we will replace with pandas' NA.

# In[ ]:


# Extract all text blocks enclosed by <record> and </record> tags from combined_text
# This captures every record, which is a text version
records = re.findall(r'<record>(.*?)</record>', combined_text)

# Define a helper function to parse features within each record
# The function extracts specific information enclosed by different tags within each record
def parse_record(record_text):
    data = {} 
    # Extract gmap_id from the record
    data['gmap_id'] = re.search(r'<gmap_id>(.*?)</gmap_id>', record_text).group(1)
    # Extract user_id from the record
    data['user_id'] = re.search(r'<user_id>(.*?)</user_id>', record_text).group(1)
    # Extract name from the record
    data['name']    = re.search(r'<name>(.*?)</name>', record_text).group(1)
    # Extract time from the record
    data['time']    = re.search(r'<time>(.*?)</time>', record_text).group(1)
    # Extract rating from the record
    data['rating']  = re.search(r'<rating>(.*?)</rating>', record_text).group(1)
    # Extract text (review content) from the record
    data['text']    = re.search(r'<text>(.*?)</text>', record_text).group(1)
    # Extract pics (pictures information) from the record
    data['pics']    = re.search(r'<pics>(.*?)</pics>', record_text).group(1)
    # Extract resp (response information) from the record
    data['resp']    = re.search(r'<resp>(.*?)</resp>', record_text).group(1)
    
    return data

# Parse each record and store in a list
# Each element (record) in the list is a dictionary containing parsed data from a single record
txt_data = [parse_record(record) for record in records]

# Convert the list of parsed records into a pd.DataFrame
# Replace 'None' string values with pandas's NA
txt_data = pd.DataFrame(txt_data).replace('None', pd.NA)

# Display the first 5 rows
print(f'The number of records now we have in .txt files: {len(txt_data)}')
txt_data.head(5)


# <div class="alert alert-block alert-info">
#     
# ### 4.2 Parsing `.xlsx` data <a class="anchor" name="Read"></a>

# In Section 3.2, we found that there are entire rows and columns with null values in the `.xlsx` data.
# 
# Here, we remove empty rows and columns from each sheet, and then combine all the sheets into one DataFrame. We still use `'Sheet0'` as an example for display.

# In[ ]:


# Define the column names to be retained in each sheet (to remove empty columns)
col_names = ["gmap_id", "user_id", "name", "time", "rating", "text", "pics", "resp"]

# Get the names of all sheets in the .xlsx file
sheet_names = list(excel_sheets.keys())

# Iterate over each sheet
for sheet_name in sheet_names:
    excel_sheets[sheet_name] = (excel_sheets[sheet_name]
        .loc[:, col_names]  # Select only the specified columns
        .dropna(how="all")  # Drop the empty rows
    )

# Display the first 5 rows of the sheet "Sheet0"
excel_sheets["Sheet0"].head(5)


# Combine all sheets into a single DataFrame.

# In[ ]:


# Combine all the sheets from the .xlsx sheets into a single one
excel_data = pd.concat(excel_sheets.values(), ignore_index=True)

# Display the first 5 rows from combining
print(f'The number of records now we have in .xlsx file: {len(excel_data)}')
excel_data.head(5)


# <div class="alert alert-block alert-info">
#     
# ### 4.3 Combining DataFrames Together <a class="anchor" name="latin"></a>

# Concatenate the DataFrames converted from `.txt` and `.xlsx` files vertically into one DataFrame, named `review_info`.

# In[ ]:


# Concatenate the DataFrame 'txt_data' and the combined Excel data (excel_data) vertically
# Combines the data from the .txt files and the .xlsx file into a single DataFrame
review_info = pd.concat([txt_data, excel_data], ignore_index=True)

# Display the first 5 rows of the combined DataFrame
print(f'The number of records in the combined DataFrame: {len(review_info)}')
review_info.head(5)


# -------------------------------------

# <div class="alert alert-block alert-warning"> 
# 
# ## 5 Data Processing <a class="anchor" name="load"></a>
# 
# </div>

# Now we have the complete DataFrame, let's start processing it to achieve the required content and format.

# <div class="alert alert-block alert-info">
#     
# ### 5.1 Creating the Required Columns <a class="anchor" name="latin"></a>

# Create two new columns, `If_response` and `If_pic`, to indicate whether there is a response and if there are pictures.
# 
# If the original column value is empty, the new column value is `'N'`, otherwise it is `'Y'`.

# In[ ]:


# The lambda function checks if the `resp` column is NaN. If it is NaN, it assigns 'N'. Otherwise, it assigns 'Y'.
review_info["If_response"] = review_info["resp"].apply(lambda x: 'N' if pd.isna(x) else 'Y')

# The lambda function checks if the `resp` column is NaN. If it is NaN, it assigns 'N'. Otherwise, it assigns 'Y'.
review_info["If_pic"] = review_info["pics"].apply(lambda x: 'N' if pd.isna(x) else 'Y')

# Display the updated DataFrame with the new "If_response" and "If_pic" columns
review_info[['resp', 'If_response', 'pics', 'If_pic']].head(5)


# Define a helper function to create the `pic_dim` column. The regex pattern:
# 
# - `'=w(\d+)-h(\d+)-k-no-p'` describes the context which the dimensions of the pictures are specified in.
# - `'w(\d+)'` describes the width, and `'h(\d+)'` describes the height of the picture.

# In[ ]:


# Define a helper function to extract dimensions (height and width) from the `pics` column
def extract_dimensions_from_pics(pictures):
    # Initialize an empty list to store the dimensions
    dimensions = []  
    
    # If the `pictures` value is NaN, return the empty list
    if pd.isna(pictures):
        return dimensions
    
    # Search for patterns in the 'pictures' string that describe the image dimensions
    matches = re.findall(r'=w(\d+)-h(\d+)-k-no-p', pictures)
    
    # Iterate over all matched dimension strings
    for match in matches:
        width, height = map(int, match)   # Convert the captured width and height to integers
        dimension = [height, width]   # Store dimensions as a list with height & width
        dimensions.append(dimension)   # Append the dimension (of a single picture) to the dimension list
    
    return dimensions

# Apply the helper function to the `pics` column and create the new column `pic_dim`
review_info["pic_dim"] = review_info["pics"].apply(extract_dimensions_from_pics)

# Display the first 5 rows, showing the relevant columns
review_info[['resp', 'If_response', 'pics', 'If_pic', 'pic_dim']].head(5)


# <div class="alert alert-block alert-info">
#     
# ### 5.2 Coverting Data Type <a class="anchor" name="latin"></a>

# Rename certain columns.

# In[ ]:


review_info = review_info.rename(columns={"rating": "review_rating", "text": "review_text"})
review_info.head(5)


# Convert the `time` column, which is in ms unit, into UTC time in `YYYY-MM-DD HH:mm:ss` format.

# In[ ]:


# 'unit="ms"' specifies that the input time is in unit of ms
# 'utc=True' ensures that the time is treated as UTC
# 'dt.floor('s')' rounds down to the nearest second (to remove ms information)
# 'dt.tz_localize(None)' removes the timezone information
review_info["time"] = pd.to_datetime(review_info["time"], unit="ms", utc=True).dt.floor('s').dt.tz_localize(None)

review_info.head(5)


# Unify the `review_rating` column into integer type.

# In[ ]:


review_info['review_rating'] = review_info['review_rating'].astype(float)
review_info.head(5)


# In[ ]:


# temp
review_info.dtypes


# <div class="alert alert-block alert-info">
#     
# ### 5.3 Foucus on `text` Column <a class="anchor" name="latin"></a>

# The following records reflect several issues in the `review_text` column:
# 1. There are duplicate rows
# 2. Emojis are present
# 3. There are `\n` (found in the data imported from the `.xlsx` file)
# 4. Non-English languages are present

# In[ ]:


review_info.loc[review_info.user_id.isin(["106628614370370530927", "114818512343072076712", "103103435538623129771"])]


# Non-English `review_text` values are found including google translation text and the original language text.

# In[ ]:


review_info.loc[review_info.user_id == "103103435538623129771", "review_text"][35069]


# Start cleaning.
# 
# Let's delete the duplicate rows first. We believe that the columns `gmap_id`, `user_id`, and `time` can specify each row, that's why we set a `subset` parameter in this way.

# In[ ]:


review_info = review_info.drop_duplicates(subset=["gmap_id", "user_id", "time"])


# Convert the UTF-8 formatted range of emojis to be removed into a regex, and then remove all these emojis from the `review_text` column.

# In[ ]:


# Define a regex pattern to match various emojis (given by uni)
emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)

# Apply the emoji removing pattern to the 'review_text' column
# The lambda function checks if the value is a string
review_info['review_text'] = review_info['review_text'].apply(lambda x: emoji_pattern.sub(r'', x) if isinstance(x, str) else x)


# Remove all `\n` from the `review_text` column. These `\n` come from the `.xlsx` data.

# In[ ]:


review_info['review_text'] = review_info['review_text'].str.replace('\n', ' ')


# For non-English reviews, extract the Google translation and remove the rest of the text.

# In[ ]:


# Define a helper function to extract the translated text from review_text if it exists
def extract_translated_text(text):
    if isinstance(text, str):  # Check if the input is a string (it may be a None value)
        # Search for the translated text within the review_text
        # The pattern looks for text between "(Translated by Google)" and "(Original)"
        match = re.search(r'\(Translated by Google\)(.*?)\(Original\)', text)
        if match:
            # If a match is found, return the translated text, removing any leading or trailing whitespace
            return match.group(1).strip()
        else:
            # If no match is found, return the original text
            return text
    else:
        # If the input is not a string, return the original value (None value)
        return text

# Apply the extract_translated_text function to the review_text column
review_info['review_text'] = review_info['review_text'].apply(extract_translated_text)


# Convert the values in the `review_text` column to lowercase.

# In[ ]:


review_info['review_text'] = review_info['review_text'].str.lower()


# Check the effect of the cleaning process.
# 
# Please note that the above operations on the `review_text` column may turn some values into `""`. Based on the clarification on the ED Forum, we decided not to change them to `None`.

# In[ ]:


review_info.loc[review_info.user_id.isin(["106628614370370530927", "114818512343072076712", "103103435538623129771"])]


# In[ ]:


review_info.loc[review_info.user_id == "103103435538623129771", "review_text"][35069]


# <div class="alert alert-block alert-warning"> 
# 
# ## 6 Generate Output Files <a class="anchor" name="write"></a>
# 
# </div>

# The content and format in the DataFrame are ready. Now it's time for the final step: generating the required output files.

# <div class="alert alert-block alert-info">
#     
# ### 6.1 Writing to `.csv` File <a class="anchor" name="test_xml"></a>

# Using the processed DataFrame `review_info`, we group the data by `gmap_id` and calculate the required columns `review_count`, `review_text_count`, and `response_count`. Then, output the results as a `.csv` file.

# In[ ]:


# Calculate the review_count, review_text_count, and response_count for each gmap_id
csv_output = review_info.groupby('gmap_id').agg(
    review_count=('gmap_id', 'size'),  # Count the total number of reviews for each gmap_id
    review_text_count=('review_text', 'count'),  # Count the number of not None review_text for each gmap_id
    response_count=('resp', 'count')  # Count the number of not None responses for each gmap_id
).reset_index()

# Save the resulting DataFrame as a .csv file
csv_output.to_csv('task1_020.csv', index=False)

# Display the aggregating data
csv_output


# <div class="alert alert-block alert-info">
#     
# ### 6.2 Writing to `.json` File <a class="anchor" name="test_xml"></a>

# Convert all null values to `"None"`. 
# 
# In the `.json` sample output, null values are represented as `"None"`, instead of `NaN`, `<NA>`, or `null`.

# In[ ]:


checked_columns = ["gmap_id", "user_id", "name", "time", "review_rating", "review_text", "pics", "resp"]
review_info[checked_columns] = review_info[checked_columns].applymap(lambda x: "None" if pd.isna(x) else x)


# Following the format of the sample output:
# 
# * In an initial dictionary, each `gmap_id` is treated as a key, and the corresponding value is a dictionary. 
# * This dictionary contains 3 keys: `reviews`, `earliest_review_date`, and `latest_review_date`. 
# * Among the 3 keys, the value of `reviews` is a list. 
# * Each element in the list is a dictionary that stores the 7 keys `user_id`, `time`, `review_rating`, `review_text`, `if_pic`, `pic_dim`, and `if_response`, along with their corresponding values.

# In[ ]:


# Initialize an empty dictionary to store the `.json` structure
json_output = {}

# Group the DataFrame by 'gmap_id'
groups = review_info.groupby('gmap_id')

# Iterate over each group corresponding to a unique 'gmap_id'
for gmap_id, group in groups:
    reviews = []  # Initialize a list to store reviews for the current 'gmap_id'

    # Iterate over each row in the group
    for _, row in group.iterrows():
        # Convert pic_dim to a string format list
        pic_dim = [[str(h), str(w)] for h, w in row["pic_dim"]]

        # Create a dictionary representing a single review
        review = {
            "user_id": row["user_id"],
            "time": row["time"].strftime('%Y-%m-%d %H:%M:%S'),  # Format time as a string in 'YYYY-MM-DD HH:MM:SS'
            "review_rating": row["review_rating"],
            "review_text": row["review_text"],
            "if_pic": row["If_pic"], 
            "pic_dim": pic_dim,
            "if_response": row["If_response"]
        }
        reviews.append(review)  # Add the review to the list of reviews

    # Find the earliest and latest review dates for the current 'gmap_id'
    earliest_review_date = group["time"].min().strftime('%Y-%m-%d %H:%M:%S')
    latest_review_date = group["time"].max().strftime('%Y-%m-%d %H:%M:%S')

    # Construct the JSON structure for the current 'gmap_id'
    json_output[gmap_id] = {
        "reviews": reviews,  # Include the list of reviews
        "earliest_review_date": earliest_review_date,  # Include the earliest review date
        "latest_review_date": latest_review_date  # Include the latest review date
    }

# Save the JSON data to a file, ensuring proper formatting
with open('task1_020.json', 'w') as f:
    json.dump(json_output, f, indent=2)  # Write the JSON data to a file with indentation for readability
    
json_output


# <div class="alert alert-block alert-info">
#     
# ### 6.3 Verification of the Generated Files <a class="anchor" name="test_xml"></a>

# The `.csv` and `.json` files generated in sections 6.1 and 6.2 have passed the `task1_test.py`.

# -------------------------------------

# <div class="alert alert-block alert-warning"> 
# 
# ## 7 Summary <a class="anchor" name="summary"></a>
# 
# </div>

# At this point, we have extracted the information from the raw files and generated output files ready for the next steps of processing and analysis.

# -------------------------------------

# <div class="alert alert-block alert-warning"> 
# 
# ## 8 References <a class="anchor" name="Ref"></a>
# 
# </div>

# [1]<a class="anchor" name="ref-2"></a> pandas.to_datetime, https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_datetime.html, Accessed 12/08/2022.
