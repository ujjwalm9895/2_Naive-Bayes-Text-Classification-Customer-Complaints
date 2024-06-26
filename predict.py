import os  
import config 
import argparse 
from Source.utils import load_file  

from nltk.corpus import stopwords  # Import stopwords from NLTK for text preprocessing.
from nltk.tokenize import word_tokenize, RegexpTokenizer  # Import word_tokenize and RegexpTokenizer for text tokenization.

# Define a list of English stopwords and a tokenizer to remove punctuation.
sw = stopwords.words('english')
tokenizer = RegexpTokenizer(r'\w+')

def main(args):
    # Load the vectorizer and the model from saved files.
    vect = load_file(os.path.join(config.output_folder, config.vect_file))
    model = load_file(os.path.join(config.output_folder, config.model_file))

    test_complaints = [args.test_complaint]

    # Convert text to lowercase.
    test_complaints = [r.lower() for r in test_complaints]

    # Tokenize the text.
    test_tokens = [word_tokenize(r) for r in test_complaints]

    # Remove stop words from tokens.
    test_tokens = [[word for word in t if word not in sw] for t in test_tokens]

    # Remove punctuation from tokens.
    test_tokens = [["".join(tokenizer.tokenize(word)) for word in t if len(tokenizer.tokenize(word)) > 0] for t in test_tokens]

    # Remove specific tokens like 'xxxx' and '000'.
    test_tokens = [[t for t in token if t not in ['xxxx', '000']] for token in test_tokens]

    # Join tokens to form cleaned test complaints.
    clean_test_complaints = [" ".join(complaint) for complaint in test_tokens]

    # Vectorize the cleaned test complaints using the same vectorizer.
    X_test = vect.transform(clean_test_complaints)

    # Make predictions using the loaded model.
    test_prediction = model.predict(X_test)[0]
    print(f"Prediction: {test_prediction}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_complaint", type=str, help="Input file name")
    args = parser.parse_args()
    main(args)
