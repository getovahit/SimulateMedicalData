import pandas as pd
import numpy as np
from sdv.tabular import CTGAN
from sdv.evaluation import evaluate
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

# CTGAN Parameters - Adjust these as needed
EPOCHS = 300
BATCH_SIZE = 500
GENERATOR_DIM = (256, 256)
DISCRIMINATOR_DIM = (256, 256)
GENERATOR_LR = 2e-4
DISCRIMINATOR_LR = 2e-4
DISCRIMINATOR_STEPS = 1
LOG_FREQUENCY = True
VERBOSE = False
EMBEDDING_DIM = 128
COMPRESS_DIMS = (128, 128)
DECOMPRESS_DIMS = (128, 128)
CUDA = True

# Load the data
data = pd.read_csv('your_data.csv', sep='\t')

# Remove identifier column but keep split columns
columns_to_remove = ['eid']
data = data.drop(columns=columns_to_remove)

# Handle missing values
numeric_columns = data.select_dtypes(include=[np.number]).columns
categorical_columns = data.select_dtypes(exclude=[np.number]).columns.drop(['set/split', 'finalsplit'])

for col in numeric_columns:
    data[col].fillna(data[col].median(), inplace=True)

for col in categorical_columns:
    data[col].fillna(data[col].mode()[0], inplace=True)

# Encode categorical variables
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
encoded_cats = encoder.fit_transform(data[categorical_columns])
encoded_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(categorical_columns))

# Combine encoded categorical variables with numeric variables and split columns
final_data = pd.concat([data[numeric_columns], encoded_df, data[['set/split', 'finalsplit']]], axis=1)

# Split the data based on the 'set/split' column
train_data = final_data[final_data['set/split'] == 'train']
test_data = final_data[final_data['set/split'] == 'test']

# Remove split columns before training
train_data = train_data.drop(columns=['set/split', 'finalsplit'])
test_data = test_data.drop(columns=['set/split', 'finalsplit'])

# Initialize and fit the CTGAN model with custom parameters
model = CTGAN(
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    generator_dim=GENERATOR_DIM,
    discriminator_dim=DISCRIMINATOR_DIM,
    generator_lr=GENERATOR_LR,
    discriminator_lr=DISCRIMINATOR_LR,
    discriminator_steps=DISCRIMINATOR_STEPS,
    log_frequency=LOG_FREQUENCY,
    verbose=VERBOSE,
    embedding_dim=EMBEDDING_DIM,
    compress_dims=COMPRESS_DIMS,
    decompress_dims=DECOMPRESS_DIMS,
    cuda=CUDA
)

model.fit(train_data)

# Generate synthetic data
num_samples = len(train_data)
synthetic_data = model.sample(num_samples)

# Evaluate the synthetic data using SDV's evaluate function
evaluation_results = evaluate(synthetic_data, train_data)
print("SDV Evaluation Results:")
print(evaluation_results)

# Define test functions
def compare_distributions(real_data, synthetic_data, column):
    plt.figure(figsize=(10, 6))
    sns.kdeplot(real_data[column], label='Real')
    sns.kdeplot(synthetic_data[column], label='Synthetic')
    plt.title(f'Distribution Comparison: {column}')
    plt.legend()
    plt.savefig(f'distribution_comparison_{column}.png')
    plt.close()

def compare_correlations(real_data, synthetic_data):
    real_corr = real_data.corr()
    synth_corr = synthetic_data.corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(real_corr - synth_corr, cmap='coolwarm', center=0)
    plt.title('Correlation Difference (Real - Synthetic)')
    plt.savefig('correlation_difference.png')
    plt.close()

def machine_learning_utility_test(real_data, synthetic_data, target_column):
    # Prepare real data
    X_real = real_data.drop(target_column, axis=1)
    y_real = real_data[target_column]
    X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(X_real, y_real, test_size=0.2, random_state=42)

    # Train on real, test on real
    model_real = RandomForestClassifier(random_state=42)
    model_real.fit(X_train_real, y_train_real)
    y_pred_real = model_real.predict(X_test_real)
    real_accuracy = accuracy_score(y_test_real, y_pred_real)
    real_f1 = f1_score(y_test_real, y_pred_real, average='weighted')

    # Train on synthetic, test on real
    X_synth = synthetic_data.drop(target_column, axis=1)
    y_synth = synthetic_data[target_column]
    model_synth = RandomForestClassifier(random_state=42)
    model_synth.fit(X_synth, y_synth)
    y_pred_synth = model_synth.predict(X_test_real)
    synth_accuracy = accuracy_score(y_test_real, y_pred_synth)
    synth_f1 = f1_score(y_test_real, y_pred_synth, average='weighted')

    print(f"Real Data - Accuracy: {real_accuracy:.4f}, F1 Score: {real_f1:.4f}")
    print(f"Synthetic Data - Accuracy: {synth_accuracy:.4f}, F1 Score: {synth_f1:.4f}")

# Perform tests
print("\nPerforming distribution comparisons...")
for column in train_data.columns:
    if train_data[column].dtype in ['int64', 'float64']:
        compare_distributions(train_data, synthetic_data, column)

print("Performing correlation comparison...")
compare_correlations(train_data, synthetic_data)

print("\nPerforming machine learning utility test...")
# Assuming 'AD' is a binary target column for this example
# Replace 'AD' with an appropriate target column from your dataset
machine_learning_utility_test(train_data, synthetic_data, 'AD')

# Add back the split columns to the synthetic data
synthetic_data['set/split'] = 'synthetic'
synthetic_data['finalsplit'] = 'synthetic'

# Save the synthetic data
synthetic_data.to_csv('synthetic_medical_data.csv', index=False)

# Save the model for future use
model.save('ctgan_medical_model.pkl')

# Optional: Generate synthetic test data
synthetic_test_data = model.sample(len(test_data))
synthetic_test_data['set/split'] = 'synthetic_test'
synthetic_test_data['finalsplit'] = 'synthetic_test'

# Combine all datasets
all_data = pd.concat([final_data, synthetic_data, synthetic_test_data], axis=0)

# Save the combined dataset
all_data.to_csv('combined_medical_data.csv', index=False)

print("\nProcess completed. Check the generated CSV files and PNG images for results.")
