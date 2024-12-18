PBRTQC - Data Analysis and Visualization for Systematic Error Detection in Clinical Laboratories
This project includes a Python-based application named PBRTQC, which is designed for data analysis, statistical modeling, and visualization. Users can load their own datasets and test the application functionalities.

Important Note
This project is actively maintained and improved. Code comments and repetitive sections will be periodically reviewed and updated to enhance clarity and performance.

Features
Data analysis for different patient categories (ICU, inpatient clinics, emergency, etc.).
Data cleaning, transformation, and statistical control limit determination.
Automatic PBRTQC processes and customizable visualizations.
User-friendly interface for easy interaction.
Supports multiple data formats (CSV, Excel).
Setup
Required Libraries
Ensure you have Python 3.x installed and the following libraries:

numpy
pandas
matplotlib
scipy
PyQt5
tqdm
joblib
filelock
Install all dependencies using the following command:

pip install -lib

Additional Requirements
Place the Analytes Indices.xlsx file in the project directory. This file is essential for data processing.
Use the "Select Excel/CSV File" button on the interface to load your data file.
Supported formats: CSV, Excel.
Set Parameters
Adjust patient category, analysis parameters, and conversion factors as needed.
Perform Analysis and Visualization
Start the automatic PBRTQC process using the "Auto PBRTQC" button.
Select visualization options from the right-hand panel.

If you encounter any issues or have suggestions, feel free to open an issue. Contributions are welcome, and we appreciate all feedback!
