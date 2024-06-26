import os  
import config  
from Source.utils import save_file  
from sklearn.metrics import accuracy_score  
from sklearn.naive_bayes import MultinomialNB 

def train_model(X_train, X_test, y_train, y_test):
    # Create a Multinomial Naive Bayes model.
    model = MultinomialNB()
    
    # Print a message indicating that the model training is in progress.
    print("Training the model...")
    
    # Train the model using the training data.
    model.fit(X_train, y_train)
    
    # Save the trained model object to a file.
    save_file(os.path.join(config.output_folder, config.model_file), model)
    
    # Make predictions on the test set.
    test_pred = model.predict(X_test)
    
    # Calculate the test accuracy by comparing predicted labels with actual labels.
    test_accuracy = accuracy_score(y_test, test_pred)
    
    # Return the test accuracy.
    return test_accuracy
