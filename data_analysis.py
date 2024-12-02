import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(page_title="Data Analysis Tool", layout="wide")

# Main title
st.title("Data Analysis Tool")

def get_friendly_dtype(series):
    """Convert pandas dtype to user-friendly string"""
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        if pd.api.types.is_integer_dtype(series):
            return "Integer"
        else:
            return "Decimal Number"
    elif pd.api.types.is_bool_dtype(series):
        return "Boolean"
    elif pd.api.types.is_datetime64_any_dtype(series):
        return "Date/Time"
    elif pd.api.types.is_categorical_dtype(series):
        return "Categorical"
    else:
        return "Text"

def create_column_anchor(column_name):
    """Create a cleaned anchor name for a column"""
    return f"col_{column_name.replace(' ', '_').replace('[', '_').replace(']', '_').replace('(', '_').replace(')', '_')}"

def format_count_with_percentage(count, total):
    """Format count with percentage"""
    percentage = (count / total * 100) if total > 0 else 0
    return f"{count:,} ({percentage:.1f}%)"

def get_percentiles(series):
    """Calculate key percentiles for a numeric series"""
    total_count = len(series)
    
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        return {
            'min': series.min(),
            '10th': series.quantile(0.10),
            '25th': series.quantile(0.25),
            '50th (median)': series.quantile(0.50),
            '75th': series.quantile(0.75),
            '99th': series.quantile(0.99),
            'max': series.max()
        }
    elif pd.api.types.is_bool_dtype(series):
        true_count = series.sum()
        false_count = (~series).sum()
        return {
            'True': format_count_with_percentage(true_count, total_count),
            'False': format_count_with_percentage(false_count, total_count)
        }
    else:  # Handle string/categorical data
        value_counts = series.value_counts()
        stats = {
            'Total Values': format_count_with_percentage(total_count, total_count),
            'Unique Values': format_count_with_percentage(len(value_counts), total_count),
        }
        
        # Add most common values if they exist
        if len(value_counts) > 0:
            stats['Most Common'] = str(value_counts.index[0])
            stats['Most Common Count'] = format_count_with_percentage(value_counts.iloc[0], total_count)
        
        if len(value_counts) > 1:
            stats['Second Most Common'] = str(value_counts.index[1])
            stats['Second Most Count'] = format_count_with_percentage(value_counts.iloc[1], total_count)
            
        return stats

def analyze_column_distribution(df, column):
    """Analyze distribution of a column"""
    series = df[column]
    total_count = len(series)
    
    st.subheader(f"Analysis of {column}")
    
    # Display data type
    st.write(f"**Data Type:** {get_friendly_dtype(series)}")
    
    # Missing values information
    missing_count = series.isna().sum()
    valid_count = total_count - missing_count
    
    st.write("**Data Completeness:**")
    st.write(f"- Valid values: {format_count_with_percentage(valid_count, total_count)}")
    if missing_count > 0:
        st.write(f"- Missing values: {format_count_with_percentage(missing_count, total_count)}")
    
    # Create distribution plot
    if pd.api.types.is_bool_dtype(series):
        # Handle boolean columns
        counts = series.value_counts()
        fig = px.bar(x=['True', 'False'], 
                    y=[counts.get(True, 0), counts.get(False, 0)],
                    title=f"Distribution of {column}")
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
    elif pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        # Handle numeric (non-boolean) columns
        fig = px.histogram(df, x=column, marginal="box")
        fig.update_layout(height=300, title=f"Distribution of {column}")
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        # For categorical data, show bar chart
        value_counts = series.value_counts()
        fig = px.bar(x=value_counts.index[:20], 
                    y=value_counts.values[:20],
                    title=f"Top 20 Values in {column}")
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # Display statistics
    stats = get_percentiles(series.dropna())  # Remove NA values for statistics
    col1, col2, col3 = st.columns(3)
    
    # Split stats into three columns
    stats_items = list(stats.items())
    items_per_col = (len(stats_items) + 2) // 3
    
    with col1:
        for k, v in stats_items[:items_per_col]:
            if isinstance(v, (int, float)):
                st.write(f"{k}: {v:,.2f}")
            else:
                st.write(f"{k}: {v}")
    with col2:
        for k, v in stats_items[items_per_col:2*items_per_col]:
            if isinstance(v, (int, float)):
                st.write(f"{k}: {v:,.2f}")
            else:
                st.write(f"{k}: {v}")
    with col3:
        for k, v in stats_items[2*items_per_col:]:
            if isinstance(v, (int, float)):
                st.write(f"{k}: {v:,.2f}")
            else:
                st.write(f"{k}: {v}")
    
    # Value counts and unique values section
    value_counts = series.value_counts()
    
    # Create two columns for the table and pie chart
    table_col, pie_col = st.columns([3, 2])
    
    with table_col:
        # Display value counts in a dataframe
        value_counts_df = pd.DataFrame({
            'Value': value_counts.index,
            'Count': value_counts.values,
            'Percentage': value_counts.values / total_count * 100
        })
        value_counts_df['Display'] = value_counts_df.apply(
            lambda x: f"{int(x['Count']):,} ({x['Percentage']:.1f}%)", axis=1)
        
        st.dataframe(
            value_counts_df[['Value', 'Display']].rename(columns={'Display': 'Count (Percentage)'}),
            height=200
        )
    
    with pie_col:
        # Create pie chart
        # If there are too many values, only show top 10 and group others
        if len(value_counts) > 10:
            top_values = value_counts.head(10)
            others_sum = value_counts[10:].sum()
            pie_values = pd.concat([top_values, pd.Series({'Others': others_sum})])
            
            # Calculate percentages for labels
            pie_percentages = (pie_values / pie_values.sum() * 100).round(1)
            pie_labels = [f"{label} ({value:,}, {percentage}%)" 
                         for label, value, percentage 
                         in zip(pie_values.index, pie_values.values, pie_percentages)]
        else:
            pie_values = value_counts
            pie_percentages = (pie_values / pie_values.sum() * 100).round(1)
            pie_labels = [f"{label} ({value:,}, {percentage}%)" 
                         for label, value, percentage 
                         in zip(pie_values.index, pie_values.values, pie_percentages)]
        
        fig = go.Figure(data=[go.Pie(
            labels=pie_labels,
            values=pie_values,
            hole=0.3,
            textinfo='none',  # Don't show text on pie slices
            showlegend=True,  # Show legend instead
        )])
        
        fig.update_layout(
            title=f"Value Distribution",
            height=400,
            legend=dict(
                yanchor="top",
                y=1.0,
                xanchor="left",
                x=1.0
            )
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")  # Add a separator between columns

# File upload
uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Load data based on file type
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:  # Excel file
            # Add Excel sheet selector if the file is Excel
            excel_file = pd.ExcelFile(uploaded_file)
            if len(excel_file.sheet_names) > 1:
                sheet_name = st.selectbox("Select sheet:", excel_file.sheet_names)
            else:
                sheet_name = excel_file.sheet_names[0]
            df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
        
        # Dataset Overview
        st.header("Dataset Overview")
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isna().sum().sum()
        duplicate_rows = df.duplicated().sum()
        
        # Display dataset metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rows", f"{df.shape[0]:,}")
        with col2:
            st.metric("Columns", f"{df.shape[1]:,}")
        with col3:
            st.metric("Missing Cells", 
                     format_count_with_percentage(missing_cells, total_cells))
        with col4:
            st.metric("Duplicate Rows", 
                     format_count_with_percentage(duplicate_rows, df.shape[0]))
        
        # Column Types Overview
        st.subheader("Column Types Overview")
        type_counts = pd.Series({col: get_friendly_dtype(df[col]) for col in df.columns}).value_counts()
        st.write("Number of columns by type:")
        for dtype, count in type_counts.items():
            st.write(f"- {dtype}: {count}")
        
        # Data Preview (100 rows)
        st.subheader("Data Preview (100 rows)")
        st.dataframe(df.head(100))
        
        # Add column navigation section
        st.subheader("Jump to Column")
        
        # Create columns for the navigation buttons
        num_cols = 4  # Number of columns in the layout
        cols = st.columns(num_cols)
        
        # Create navigation buttons
        for idx, column in enumerate(df.columns):
            col_idx = idx % num_cols
            with cols[col_idx]:
                dtype = get_friendly_dtype(df[column])
                st.markdown(f"[{column}](#analysis-of-{column.lower()}) ({dtype})")
        
        # Add some space after the navigation
        st.markdown("---")
        
        # Column-by-column analysis
        st.header("Column Analysis")
        
        # Analyze each column
        for column in df.columns:
            # Create an anchor for this column
            st.markdown(f"<div id='analysis-of-{column.lower()}'></div>", unsafe_allow_html=True)
            analyze_column_distribution(df, column)
    
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        st.info("Please make sure your file is properly formatted and try again.")

else:
    st.info("Please upload a CSV or Excel file to begin analysis")