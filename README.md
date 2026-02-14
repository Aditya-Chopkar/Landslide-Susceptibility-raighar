# Landslide Susceptibility Analysis

This repository provides tools and methods for performing landslide susceptibility analysis in the Raighar region. Follow the instructions below to set up the environment and run the analysis.

## Table of Contents
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Running the Analysis](#running-the-analysis)
- [Results Interpretation](#results-interpretation)
- [Contributing](#contributing)

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Aditya-Chopkar/Landslide-Susceptibility-raighar.git
   cd Landslide-Susceptibility-raighar
   ```

2. **Install Required Packages**
   It is recommended to create a virtual environment for this project:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
   Then, install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Data Preparation

Before running the analysis, ensure that you have the necessary datasets. Place your datasets in the `data/` directory. The required files include:
- **geographical_data.csv**: Contains geographical features relevant to landslide susceptibility.
- **historical_landslides.csv**: Contains records of past landslides in the region.

Ensure that the data files are formatted correctly as per the specifications provided in the project documentation.

## Running the Analysis

Once the data is prepared, you can run the landslide susceptibility analysis as follows:

```bash
python analysis_script.py --data_path data/geographical_data.csv --historical_data_path data/historical_landslides.csv
```

Make sure to replace the paths with the correct ones if you have named your files differently.

## Results Interpretation

The results will be saved in the `results/` directory. You will find detailed outputs including:
- Susceptibility maps
- Statistical analysis and summary

Refer to the documentation for additional details on interpreting the results.

## Contributing

If you wish to contribute to the project, please fork the repository and submit a pull request with your changes. For major changes, please open an issue first to discuss what you would like to change.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

For further assistance or inquiries, please contact the repository owner.