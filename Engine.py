import os  
import config  
import argparse 
from Source.model import train_model  
from Source.processing import process_text 

# Define the main function that performs data processing and model training
def main(args_):
    # Create a file path by joining the input folder and file name
    file_path = os.path.join(args_.input_path, args_.file_name)

    # Process the data using the process_text function
    X_train, X_test, y_train, y_test = process_text(file_path)

    # Train the model using the train_model function
    test_accuracy = train_model(X_train, X_test, y_train, y_test)

    # Print the test accuracy
    print(f"Test accuracy: {test_accuracy}")

# Entry point of the script
if __name__ == "__main__":
    # Create an argument parser for handling command-line arguments
    parser = argparse.ArgumentParser()

    # Define command-line arguments with default values and help messages
    parser.add_argument("--file_name", type=str, default=config.file_name,
                        help="Input file name")
    parser.add_argument("--input_path", type=str, default=config.input_folder,
                        help="Input folder name")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(args)
