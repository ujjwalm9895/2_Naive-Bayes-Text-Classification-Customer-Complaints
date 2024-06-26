# Set the minimum document frequency for the CountVectorizer
min_df = 200

# Set the random state for reproducibility
random_state = 42

# Define the filename for the trained model
model_file = "nb.pkl"

# Specify the column name containing the target labels in the dataset
label_col = "Product"

# Specify the input folder where the dataset is located
input_folder = "Input"

# Specify the output folder where the results will be saved
output_folder = "Output"

# Define the filename for the serialized CountVectorizer
vect_file = "count_vect.pkl"

# Define the filename of the input CSV file
file_name = "complaints.csv"

# Specify the column name containing the consumer complaints
comp_col = "Consumer complaint narrative"

# Define a mapping from product categories to shorter labels
product_map = {
    'Vehicle loan or lease': 'vehicle_loan',
    'Credit reporting, credit repair services, or other personal consumer reports': 'credit_report',
    'Credit card or prepaid card': 'card',
    'Money transfer, virtual currency, or money service': 'money_transfer',
    'virtual currency': 'money_transfer',
    'Mortgage': 'mortgage',
    'Payday loan, title loan, or personal loan': 'loan',
    'Debt collection': 'debt_collection',
    'Checking or savings account': 'savings_account',
    'Credit card': 'card',
    'Bank account or service': 'savings_account',
    'Credit reporting': 'credit_report',
    'Prepaid card': 'card',
    'Payday loan': 'loan',
    'Other financial service': 'others',
    'Virtual currency': 'money_transfer',
    'Student loan': 'loan',
    'Consumer Loan': 'loan',
    'Money transfers': 'money_transfer'
}
