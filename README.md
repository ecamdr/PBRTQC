PBRTQC - Sadi Konuk Training and Research Hospital Application
This project includes a Python-based application named PBRTQC, which is designed for data analysis, statistical modeling, and visualization. Users can load their own datasets and test the application functionalities.

Features
Data analysis for different patient categories (ICU, inpatient clinics, emergency, etc.).
Data cleaning, transformation, and statistical control limit determination.
Automatic PBRTQC processes and customizable visualizations.
User-friendly interface for easy interaction.
Supports multiple data formats (CSV, Excel).
Setup
Required Libraries
To run this project, the following Python libraries are needed:

numpy
pandas
matplotlib
scipy
PyQt5
tqdm
joblib
filelock
You can install the dependencies using the following command:

bash
Kodu kopyala
pip install -r requirements.txt
Additional Requirements
Ensure the file Analytes Indices.xlsx is placed in the project directory.
Usage
Clone the Repository:

bash
Kodu kopyala
git clone https://github.com/yourusername/project-name.git
cd project-name
Run the Application:

bash
Kodu kopyala
python PBRTQC.py
Load Data:

Use the "Select Excel/CSV File" button on the interface to load your data file.
Supported formats: CSV, Excel.
Set Parameters:

Adjust patient category, analysis parameters, and conversion factors as needed.
Perform Analysis and Visualization:

Start the automatic PBRTQC process using the "Auto PBRTQC" button.
Select visualization options from the right-hand panel.
Example Usage
plaintext
Kodu kopyala
> Data successfully loaded.
> Training dataset transformed. Comparing with test dataset...
> Training Male-to-Female Ratio: 1.25
> Training Median Age: 45
> Outputs saved successfully.
Contributing
To contribute to this project:

Fork the repository.
Create a new branch:
bash
Kodu kopyala
git checkout -b new-feature
Commit your changes:
bash
Kodu kopyala
git commit -m "Added a new feature"
Create a pull request.
