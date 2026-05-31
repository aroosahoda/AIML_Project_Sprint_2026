import streamlit as st
import pandas as pd
import numpy as np
import io


st.set_page_config(page_title="Data Quality Analyzer", layout="wide")


st.title("📊 Data Quality Report Generator")
st.write("Upload your CSV dataset to instantly analyze data quality and get cleaning recommendations.")


uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])


if uploaded_file is not None:
    
   
    df = pd.read_csv(uploaded_file)
    cleaned_df = df.copy() 
    
  
    st.subheader("1. Dataset Overview")
    st.write(f"**Total Rows:** {df.shape[0]} | **Total Columns:** {df.shape[1]}")
    st.dataframe(df.head()) 

   
    st.subheader("2. Data Quality Analysis")
    
   
    col1, col2 = st.columns(2)
    
    with col1:
       
        duplicates = df.duplicated().sum()
        st.write(f"**Duplicate Rows Detected:** {duplicates}")
        
       
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        missing_df = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
        missing_df = missing_df[missing_df['Missing Count'] > 0]
        
        st.write("**Columns with Missing Values:**")
        if not missing_df.empty:
            st.dataframe(missing_df)
        else:
            st.success("No missing values found!")

    with col2:
       
        types_df = pd.DataFrame({'Data Type': df.dtypes}).astype(str)
        st.write("**Column Data Types:**")
        st.dataframe(types_df)

    st.subheader("3. Automated Cleaning Suggestions")
    
    suggestions = []
    
    if duplicates > 0:
        suggestions.append(f"Removed {duplicates} duplicate row(s).")
        cleaned_df = cleaned_df.drop_duplicates()
        
    for col, row in missing_df.iterrows():
        if row['Missing %'] > 50:
            suggestions.append(f"Dropped column '{col}' because it has >50% missing data.")
            cleaned_df = cleaned_df.drop(columns=[col])
        else:
            suggestions.append(f"Filled missing values in '{col}' with median/mode.")
            
            if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
            else:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0])

    
    if suggestions:
        for idx, step in enumerate(suggestions, 1):
            st.write(f"{idx}. {step}")
    else:
        st.success("Your data looks perfectly clean! No actions needed.")

   
    st.subheader("4. Before vs. After Results")
    
   
    missing_before = df.isnull().any(axis=1).sum() 
    missing_after = cleaned_df.isnull().any(axis=1).sum()
    
   
    metric1, metric2 = st.columns(2)
    with metric1:
        st.metric(label="Rows with Missing Values", value=missing_before, delta=-missing_before if missing_before > 0 else 0, delta_color="inverse")
    with metric2:
        st.metric(label="Rows with Missing Values (Cleaned)", value=missing_after)

    st.write("**Cleaned Dataset Preview:**")
    st.dataframe(cleaned_df.head())
    
   
    csv_data = cleaned_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Cleaned Dataset",
        data=csv_data,
        file_name='cleaned_data.csv',
        mime='text/csv',
    )

    