# Task-5-Data-Science---Text-Classification-on-consumer-complaint-dataset

1. Introduction
   
    Consumer complaints contain valuable insights that can be used to categorize issues into predefined categories for better customer support and trend analysis. In     his project, we classify consumer complaints into four categories:

         1.Credit Reporting, Credit Repair, or Other Personal Consumer Reports
         2.Debt Collection
         3.Consumer Loan
         4.Mortgage
   
    We employ various machine learning and deep learning models to analyze, process, and classify textual data. This document outlines the key steps in the data 
    processing pipeline and evaluates different models for classification performance.


2. Explanatory Data Analysis and Feature Engineering
   
Before applying machine learning models, we analyze the dataset to understand its structure, distribution, and potential issues. Key steps include:

1.Loading the dataset: Extracting relevant columns (e.g., Product, Issue, Sub-Issue, and Consumer Complaint Narrative).
2.Checking for missing values: Identifying and handling missing or inconsistent values.
3.Class Distribution Analysis: Visualizing the number of complaints in each category using bar charts and pie charts.
4.Text Length Analysis: Checking the distribution of complaint text lengths.
5.Balancing the Dataset: Applying undersampling to ensure equal representation of categories for fair model training.
