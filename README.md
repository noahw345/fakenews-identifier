# Fake vs Real News Analysis

## Description

This project analyzes and compares fake and real news articles. It processes datasets, labels them, and provides insights through visualizations. The main goal is to understand the characteristics of fake news compared to real news.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Data Processing](#data-processing)
- [Visualization](#visualization)
- [Results](#results)

## Installation

To get started, clone the repository and install the required packages.

git clone https://github.com/noahw345/fakenews-identifier.git  
cd fakenews-identifier  
pip install -r requirements.txt

## Usage

Run the Jupyter Notebook to execute the analysis and visualizations.

jupyter notebook comparison.ipynb

## Data Processing

The data processing is handled in the `comparison.ipynb` file. The main steps include:

1. Reading CSV files containing news articles.
2. Labeling articles as 'fake' or 'real' based on the filename.
3. Concatenating the data into a single DataFrame.
4. Sampling a balanced number of fake and real articles for analysis.

## Visualization

The project includes visualizations to compare the counts of fake and real news articles. The results are displayed using Plotly and Matplotlib.

## Results

The analysis provides insights into the average length of titles for fake and real news articles. The results are visualized in bar charts.
