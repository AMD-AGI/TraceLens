import pandas as pd
import numpy as np
from openai import OpenAI
from slodels import SLAIAzureOpenAI
import os
import json
from pathlib import Path

class ExcelAnalyzer:
    def __init__(self, api_key):
        self.client = SLAIAzureOpenAI(api_key=api_key)
        self.model = "gpt-4o"
    
    def read_excel_file(self, file_path, sheet_name=None):
        """Read Excel file and return DataFrame"""
        try:
            if sheet_name:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
            else:
                df = pd.read_excel(file_path)
            print(f"Successfully loaded Excel file: {file_path}")
            print(f"Shape: {df.shape} (rows, columns)")
            return df
        except Exception as e:
            print(f"Error reading Excel file: {e}")
            return None
    
    def get_data_summary(self, df):
        """Generate a comprehensive data summary"""
        summary = {
            "shape": df.shape,
            "columns": list(df.columns),
            "data_types": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "numeric_summary": {},
            "sample_data": df.head().to_dict('records')
        }
        
        # Get numeric column statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            summary["numeric_summary"] = df[numeric_cols].describe().to_dict()
        
        return summary
    
    def create_analysis_prompt(self, data_summary, analysis_type="general"):
        """Create targeted prompts based on analysis type"""
        
        prompts = {
            "general": f"""
            I have an Excel dataset with the following characteristics:
            - Shape: {data_summary['shape'][0]} rows, {data_summary['shape'][1]} columns
            - Columns: {', '.join(data_summary['columns'])}
            - Data types: {data_summary['data_types']}
            - Missing values: {data_summary['missing_values']}
            
            Sample data (first few rows):
            {json.dumps(data_summary['sample_data'][:3], indent=2, default=str)}
            
            Please provide:
            1. A comprehensive analysis of this dataset
            2. Key insights and patterns you can identify
            3. Potential data quality issues
            4. Recommendations for further analysis
            5. Interesting questions this data could answer
            """,
            
            "business": f"""
            As a business analyst, examine this dataset:
            - Columns: {', '.join(data_summary['columns'])}
            - {data_summary['shape'][0]} records with {data_summary['shape'][1]} attributes
            
            Sample data:
            {json.dumps(data_summary['sample_data'][:3], indent=2, default=str)}
            
            Provide:
            1. Business insights and KPIs that can be derived
            2. Revenue/performance patterns if applicable
            3. Customer/market trends if relevant
            4. Actionable recommendations
            5. Risk factors or opportunities identified
            """,
            
            "statistical": f"""
            Perform statistical analysis on this dataset:
            - Shape: {data_summary['shape']}
            - Numeric columns summary: {data_summary.get('numeric_summary', {})}
            - Missing data: {data_summary['missing_values']}
            
            Please analyze:
            1. Statistical distributions and central tendencies
            2. Correlations and relationships between variables
            3. Outliers and anomalies
            4. Data completeness and quality assessment
            5. Recommended statistical tests or models
            """,
            
            "data_quality": f"""
            Assess the data quality of this dataset:
            - Columns and types: {data_summary['data_types']}
            - Missing values: {data_summary['missing_values']}
            - Sample records: {json.dumps(data_summary['sample_data'][:3], indent=2, default=str)}
            
            Evaluate:
            1. Data completeness and consistency
            2. Potential data entry errors or inconsistencies
            3. Standardization issues
            4. Duplicate detection recommendations
            5. Data cleaning and preprocessing steps needed
            """
        }
        
        return prompts.get(analysis_type, prompts["general"])
    
    def analyze_with_ai(self, prompt):
        """Send prompt to OpenAI and get analysis"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error in AI analysis: {e}"
    
    def get_sheet_names(self, file_path):
        """Get all sheet names from Excel file"""
        try:
            xl_file = pd.ExcelFile(file_path)
            return xl_file.sheet_names
        except Exception as e:
            print(f"Error reading sheet names: {e}")
            return []
    
    def analyze_excel(self, file_path, sheet_name=None, analysis_types=None):
        """Complete Excel analysis workflow"""
        if analysis_types is None:
            analysis_types = ["general", "business"]
        
        # Read the Excel file
        df = self.read_excel_file(file_path, sheet_name)
        if df is None:
            return
        
        # Generate data summary
        print("\n" + "="*50)
        print("DATA SUMMARY")
        print("="*50)
        data_summary = self.get_data_summary(df)
        
        print(f"Columns: {', '.join(data_summary['columns'])}")
        print(f"Data types: {data_summary['data_types']}")
        print(f"Missing values: {data_summary['missing_values']}")
        
        # Show sample data
        print(f"\nSample data (first 3 rows):")
        print(df.head(3).to_string())
        
        # Perform different types of analysis
        for analysis_type in analysis_types:
            print(f"\n" + "="*50)
            print(f"AI ANALYSIS - {analysis_type.upper()}")
            print("="*50)
            
            prompt = self.create_analysis_prompt(data_summary, analysis_type)
            analysis = self.analyze_with_ai(prompt)
            print(analysis)
            
            # Add separator between analyses
            print("\n" + "-"*30)
    
    def list_excel_files(self, directory="."):
        """List all Excel files in directory"""
        excel_files = []
        for file in Path(directory).glob("*.xlsx"):
            excel_files.append(file)
        for file in Path(directory).glob("*.xls"):
            excel_files.append(file)
        return excel_files

def main():  # Should be 'main()' not '__main__()'
    # Initialize the analyzer
    analyzer = ExcelAnalyzer(api_key="")
    
    # List available Excel files
    print("Available Excel files in trace-directory:")
    excel_files = analyzer.list_excel_files("./traces-diff-gpus")
    
    if not excel_files:
        print("No Excel files found in trace-directory")
        print("Please place an Excel file (.xlsx or .xls) in the trace-directory and run again.")
        return
    
    for i, file in enumerate(excel_files, 1):
        print(f"{i}. {file.name}")
    
    # Get user input for file selection
    try:
        choice = int(input(f"\nSelect file (1-{len(excel_files)}): ")) - 1
        selected_file = excel_files[choice]
    except (ValueError, IndexError):
        print("Invalid selection")
        return
    
    # Check if file has multiple sheets
    sheet_names = analyzer.get_sheet_names(selected_file)
    selected_sheet = None
    
    if len(sheet_names) > 1:
        print(f"\nAvailable sheets in {selected_file.name}:")
        for i, sheet in enumerate(sheet_names, 1):
            print(f"{i}. {sheet}")
        
        try:
            sheet_choice = int(input(f"\nSelect sheet (1-{len(sheet_names)}): ")) - 1
            selected_sheet = sheet_names[sheet_choice]
        except (ValueError, IndexError):
            print("Invalid sheet selection, using first sheet")
            selected_sheet = sheet_names[0]
    
    # Get analysis preferences
    print("\nSelect analysis types:")
    print("1. General Analysis")
    print("2. Business Analysis") 
    print("3. Statistical Analysis")
    print("4. Data Quality Assessment")
    print("5. All of the above")
    
    analysis_map = {
        "1": ["general"],
        "2": ["business"],
        "3": ["statistical"],
        "4": ["data_quality"],
        "5": ["general", "business", "statistical", "data_quality"]
    }
    
    analysis_choice = input("Enter your choice (1-5): ")
    selected_analyses = analysis_map.get(analysis_choice, ["general"])
    
    # Perform the analysis
    print(f"\nStarting analysis of {selected_file.name}")
    if selected_sheet:
        print(f"Using sheet: {selected_sheet}")
    
    analyzer.analyze_excel(
        file_path=selected_file, 
        sheet_name=selected_sheet, 
        analysis_types=selected_analyses
    )

# Run the main function
if __name__ == "__main__":
    main()
