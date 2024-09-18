# SimulateMedicalData
# Tabular GAN for Medical Data Synthesis

## Table of Contents
1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Configuration](#configuration)
6. [Output](#output)
7. [Interpretation of Results](#interpretation-of-results)
8. [Troubleshooting](#troubleshooting)
9. [Privacy and Ethical Considerations](#privacy-and-ethical-considerations)
10. [Contributing](#contributing)
11. [License](#license)

## Introduction

This project implements a Tabular Generative Adversarial Network (TGAN) for synthesizing medical data. It's designed to generate realistic, synthetic medical data based on real input data in tabular format. The primary goal is to create high-quality synthetic data that maintains the statistical properties and relationships of the original data while ensuring patient privacy.

Key features:
- Data preprocessing and encoding
- CTGAN model training and synthetic data generation
- Comprehensive quality assessment of synthetic data
- Visualization of data distributions and correlations
- Machine learning utility testing

## Prerequisites

Before you begin, ensure you have met the following requirements:
- Python 3.7 or higher
- pip (Python package installer)
- Access to a command-line interface
- Basic understanding of Python and machine learning concepts

## Installation

Follow these steps to set up the project environment:

1. Clone the repository or download the script:
   ```
   git clone [repository-url]
   cd [repository-name]
   ```
2. Automatically create the environment (if this is done, skip the next 2 steps) (only compatible with Ubuntu)
   ```
   source ./create_env.sh
   ```

3. (Optional but recommended) Create a virtual environment:
   ```
   python -m venv ganvenv
   source ganvenv/bin/activate  # On Windows, use `ganvenv\Scripts\activate`
   ```

4. Install the required packages:
   ```
   pip install -r gan_requirements.txt
   ```

   Detailed package versions:
   - pandas (1.2.0 or higher)
   - numpy (1.19.0 or higher)
   - scikit-learn (0.24.0 or higher)
   - seaborn (0.11.0 or higher)
   - matplotlib (3.3.0 or higher)
   - sdv (0.13.1 or higher)

5. If you plan to use GPU acceleration, ensure you have CUDA installed and install the appropriate PyTorch version.

## Usage

To run the script:

1. Prepare your input data:
   - Ensure your data is in CSV format with tab separation.
   - The file should include columns for 'set/split' and 'finalsplit'.
   - Remove any direct patient identifiers.

2. Update the script:
   - Open `tabular_gan_medical_data.py` in a text editor.
   - Replace `'your_data.csv'` with the path to your input data file.
   - If your target column is not 'AD', replace 'AD' in the `machine_learning_utility_test` function call with your target column name.

3. Place splits files
   - Place splits files (new_split_*) inside the same directory with the main script

4. Select phenotype
   - Change the "PHENOTYPE" variable in line 29 of the main script with the phenotype of interest to make the splits accordingly

5. Create results directory
   - Make sure there is a diretory called "distribution_comparison_Degree" in the same directory as the main script to be used for some of the results outputs, and if it doesn't exist, create one with the same name before proceeding to next step

6. Activate and run (if this is done, skip the next 2 steps) (only compatible with Ubuntu)
   - Run the following script to activate the virtual environment and use it to automatically run the script by running:
   source activate_run_env.sh

7. Activate the environment
   - activate the virtual environment by running:
   source venv/bin/activate  # On Windows, use `ganvenv\Scripts\activate`

8. Run the script:
   ```
   python tabular_gan_medical_data.py
   ```

9. Review the output files and console messages for results and any error messages.

## Configuration

You can adjust the CTGAN model parameters at the top of the script:

- `EPOCHS`: Number of training epochs
- `BATCH_SIZE`: Number of samples per batch during training
- `GENERATOR_DIM`: Dimensions of the generator network
- `DISCRIMINATOR_DIM`: Dimensions of the discriminator network
- `GENERATOR_LR` and `DISCRIMINATOR_LR`: Learning rates
- `DISCRIMINATOR_STEPS`: Number of discriminator updates per generator update
- `EMBEDDING_DIM`: Size of the random sample passed to the generator
- `COMPRESS_DIMS` and `DECOMPRESS_DIMS`: Dimensions of the encoder and decoder
- `CUDA`: Whether to use GPU acceleration (if available)

Adjust these parameters based on your data size, complexity, and computational resources.

## Output

The script generates several output files:

1. `synthetic_medical_data.csv`: The generated synthetic data
2. `combined_medical_data.csv`: Combined original and synthetic data
3. `ctgan_medical_model.pkl`: Saved CTGAN model for future use
4. Distribution comparison plots: PNG files for each numerical column
5. `correlation_difference.png`: Heatmap of correlation differences

Console output includes:
- SDV evaluation results
- Machine learning utility test results

## Interpretation of Results

1. SDV Evaluation Results:
   - Provides metrics on how well the synthetic data matches the statistical properties of the real data.
   - Look for high scores (closer to 1.0) indicating better quality.

2. Distribution Comparison Plots:
   - Compare the shapes of real and synthetic data distributions.
   - Look for similar overall shapes and ranges.

3. Correlation Difference Heatmap:
   - Areas closer to white indicate well-preserved correlations.
   - Red or blue areas show over- or under-represented correlations in synthetic data.

4. Machine Learning Utility Test:
   - Compare accuracy and F1 scores between real and synthetic data.
   - Synthetic data performance should ideally be close to real data performance.

## Troubleshooting

Common issues and solutions:

1. MemoryError: Reduce batch size or use a smaller subset of your data.
2. CUDA out of memory: Reduce model dimensions or switch to CPU (set CUDA = False).
3. Poor synthetic data quality: Try increasing epochs, adjusting network dimensions, or preprocessing data differently.

## Privacy and Ethical Considerations

While this method helps create synthetic data, it's crucial to ensure:

1. No personal identifiers are included in the input data.
2. The synthetic data doesn't inadvertently reveal real patient information.
3. Compliance with relevant data protection regulations (e.g., HIPAA, GDPR).
4. Ethical use of the synthetic data, maintaining the same standards as for real patient data.

## Contributing

Contributions to improve the script or documentation are welcome. Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature.
3. Commit your changes.
4. Push to the branch.
5. Create a new Pull Request.

## License

[Specify the license under which this project is released, e.g., MIT, Apache 2.0, etc.]

---

For any questions or issues, please [open an issue](link-to-issue-tracker) in the project repository.
