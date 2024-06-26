import os  
import config  
import pandas as pd  
from tqdm import tqdm  
from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import CountVectorizer 

from nltk.corpus import stopwords  # Import stopwords from NLTK for text preprocessing.
from nltk.tokenize import word_tokenize, RegexpTokenizer  # Import word_tokenize and RegexpTokenizer for text tokenization.

from Source.utils import save_file  # Import a custom function 'save_file' from the 'Source.utils' module.

# Define a list of English stopwords and a tokenizer to remove punctuation.
sw = stopwords.words('english')
tokenizer = RegexpTokenizer(r'\w+')

def process_text(file_path):
    # Read input data from a CSV file.
    print("Reading input data...")
    data = pd.read_csv(file_path)

    # Select the label column and text column.
    data = data[[config.label_col, config.comp_col]]

    # Drop rows with null values.
    data.dropna(inplace=True)

    # Rename the text column to "Complaint".
    data.rename({config.comp_col: "Complaint"}, axis=1, inplace=True)

    # Map product names to common names.
    data.replace({"Product": config.product_map}, inplace=True)

    # Select a subset of data (first 10,000 rows).
    data = data[1:10000]

    # Extract complaints as a list.
    complaints = list(data["Complaint"])

    # Convert text to lowercase.
    print("Converting text to lower case...")
    complaints = [c.lower() for c in tqdm(complaints)]

    # Tokenize the text.
    print("Tokenizing the text...")
    tokens = [word_tokenize(r) for r in tqdm(complaints)]

    # Remove stopwords from tokens.
    print("Removing stop words...")
    tokens = [[word for word in t if word not in sw] for t in tqdm(tokens)]

    # Remove punctuation from tokens.
    print("Removing punctuations...")
    tokens = [["".join(tokenizer.tokenize(word)) for word in t if len(tokenizer.tokenize(word)) > 0] for t in tqdm(tokens)]

    # Remove specific tokens like 'xxxx' and '000'.
    print("Removing 'xxxx' and '000' tokens...")
    tokens = [[t for t in token if t not in ['xxxx', '000']] for token in tqdm(tokens)]

    # Join tokens to form cleaned complaints.
    print("Joining tokens...")
    clean_complaints = [" ".join(complaint) for complaint in tqdm(tokens)]

    # Vectorize the cleaned text data using CountVectorizer.
    print("Vectorizing the data...")
    vect = CountVectorizer(min_df=config.min_df)
    X = vect.fit_transform(clean_complaints)
    y = data[config.label_col]

    # Split the data into training and testing sets.
    print("Split the data into train and test...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=config.random_state)

    # Save the vectorizer to a file.
    print("Saving the files...")
    save_file(os.path.join(config.output_folder, config.vect_file), vect)

    return X_train, X_test, y_train, y_test
