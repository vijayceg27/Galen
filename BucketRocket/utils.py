"""
Consolidated utilities for Text Response Analyzer
Includes database, analysis, visualization, and helper functions
"""

import os
import re
import json
import mysql.connector
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Tuple, Optional, Union
from dotenv import load_dotenv
from collections import Counter
from pathlib import Path

# ======================
# Database Utilities
# ======================

def connect_db(env_path='.env'):
    """
    Connect to MySQL database using environment variables
    
    Args:
        env_path: Path to .env file containing database credentials
        
    Returns:
        MySQL connection object
    """
    load_dotenv(dotenv_path=env_path)
    try:
        conn = mysql.connector.connect(
            host=os.getenv("DB_HOST"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_NAME")
        )
        return conn
    except mysql.connector.Error as e:
        st.error(f"Database connection error: {e}")
        raise

def fetch_projects(conn) -> pd.DataFrame:
    """
    Fetch all available projects for dropdown selection
    
    Args:
        conn: Database connection
        
    Returns:
        DataFrame with project id and name
    """
    try:
        query = "SELECT id, name FROM zoomrx.projects ORDER BY name"
        df = pd.read_sql(query, conn)
        return df
    except Exception as e:
        st.error(f"Error fetching projects: {e}")
        return pd.DataFrame()

def fetch_surveys(conn, project_id: int) -> pd.DataFrame:
    """
    Fetch surveys for a specific project
    
    Args:
        conn: Database connection
        project_id: Selected project ID
        
    Returns:
        DataFrame with survey_id and title
    """
    try:
        query = """
        SELECT ps.survey_id, lsl.surveyls_title 
        FROM zoomrx.projects_surveys ps 
        LEFT JOIN zoomrx.lime_surveys_languagesettings lsl 
            ON ps.survey_id = lsl.surveyls_survey_id 
        WHERE ps.project_id = %s
        ORDER BY lsl.surveyls_title
        """
        df = pd.read_sql(query, conn, params=[project_id])
        return df
    except Exception as e:
        st.error(f"Error fetching surveys: {e}")
        return pd.DataFrame()

def fetch_waves(conn, survey_id: int) -> pd.DataFrame:
    """
    Fetch waves for a specific survey
    
    Args:
        conn: Database connection
        survey_id: Selected survey ID
        
    Returns:
        DataFrame with wave id, start_date, and end_date
    """
    try:
        query = """
        SELECT w.id, w.start_date, w.end_date 
        FROM waves w 
        WHERE w.survey_id = %s
        ORDER BY w.start_date DESC
        """
        df = pd.read_sql(query, conn, params=[survey_id])
        return df
    except Exception as e:
        st.error(f"Error fetching waves: {e}")
        return pd.DataFrame()

def fetch_questions(conn, survey_id: str) -> List[Dict]:
    """
    Fetch open-ended questions from the survey
    
    Args:
        conn: Database connection
        survey_id: ID of the survey
        
    Returns:
        List of question dictionaries with qid, QCode, and question text
    """
    cursor = conn.cursor(dictionary=True)
    query = f"""
        SELECT lq.qid, lq.title AS QCode, lq.question
        FROM lime_questions AS lq
        WHERE lq.sid = {survey_id} AND lq.type = 'T' AND lq.parent_qid = 0
    """
    cursor.execute(query)
    return cursor.fetchall()

def fetch_responses(conn, survey_id: str, question_id: str, wave_id: int = None) -> pd.DataFrame:
    """
    Fetch responses for a specific question, optionally filtered by wave
    
    Args:
        conn: Database connection
        survey_id: ID of the survey
        question_id: ID of the question
        wave_id: Optional wave ID to filter responses
        
    Returns:
        DataFrame with id and response columns
    """
    cursor = conn.cursor(dictionary=True)
    # Properly escape the question_id for JSON path
    json_path = f'$."{ question_id}"'
    
    if wave_id:
        # Filter by specific wave
        query = f"""
            SELECT sr.id, 
                COALESCE(NULLIF(JSON_UNQUOTE(JSON_EXTRACT(sr.responses, '{json_path}')), ''), '[No Response]') AS response
            FROM survey_responses AS sr
            JOIN users_waves uw ON uw.id = sr.id
            WHERE uw.wave_id = {wave_id}
            AND JSON_UNQUOTE(JSON_EXTRACT(sr.responses, '{json_path}')) IS NOT NULL
            AND JSON_UNQUOTE(JSON_EXTRACT(sr.responses, '{json_path}')) != ''
            ORDER BY sr.id
        """
    else:
        # Use all waves for the survey (backward compatibility)
        query = f"""
            SELECT sr.id, 
                COALESCE(NULLIF(JSON_UNQUOTE(JSON_EXTRACT(sr.responses, '{json_path}')), ''), '[No Response]') AS response
            FROM survey_responses AS sr
            JOIN users_waves uw ON uw.id = sr.id
            WHERE uw.wave_id IN (SELECT id FROM waves WHERE survey_id = {survey_id})
            AND JSON_UNQUOTE(JSON_EXTRACT(sr.responses, '{json_path}')) IS NOT NULL
            AND JSON_UNQUOTE(JSON_EXTRACT(sr.responses, '{json_path}')) != ''
            ORDER BY sr.id
        """
    
    cursor.execute(query)
    responses = cursor.fetchall()
    cursor.close()
    
    df = pd.DataFrame(responses)
    
    # Filter out empty responses
    if not df.empty:
        df = df[df['response'] != '[No Response]']
        df = df[df['response'].str.strip() != '']
        df = df.dropna(subset=['response'])
    
    return df

# ======================
# Analysis Utilities
# ======================

class ResponseAnalyzer:
    """Utility class for advanced response analysis"""
    
    def __init__(self, responses_df: pd.DataFrame):
        self.responses_df = responses_df
        
    def get_response_statistics(self) -> Dict[str, Any]:
        """Calculate comprehensive response statistics"""
        if self.responses_df.empty:
            return {}
        
        responses = self.responses_df['response'].dropna()
        
        # Length statistics
        char_lengths = responses.str.len()
        word_lengths = responses.str.split().str.len()
        
        # Sentence count (approximate)
        sentence_counts = responses.str.count(r'[.!?]+')
        
        stats = {
            'total_responses': len(responses),
            'char_length': {
                'mean': char_lengths.mean(),
                'median': char_lengths.median(),
                'std': char_lengths.std(),
                'min': char_lengths.min(),
                'max': char_lengths.max()
            },
            'word_length': {
                'mean': word_lengths.mean(),
                'median': word_lengths.median(),
                'std': word_lengths.std(),
                'min': word_lengths.min(),
                'max': word_lengths.max()
            },
            'sentence_count': {
                'mean': sentence_counts.mean(),
                'median': sentence_counts.median(),
                'std': sentence_counts.std()
            }
        }
        
        return stats
    
    def extract_common_phrases(self, n: int = 10, min_words: int = 2, max_words: int = 4) -> List[Tuple[str, int]]:
        """Extract most common phrases from responses"""
        if self.responses_df.empty:
            return []
        
        responses = self.responses_df['response'].dropna()
        phrases = []
        
        for response in responses:
            words = re.findall(r'\b\w+\b', response.lower())
            for i in range(len(words)):
                for j in range(min_words, min(max_words + 1, len(words) - i + 1)):
                    phrase = ' '.join(words[i:i+j])
                    phrases.append(phrase)
        
        return Counter(phrases).most_common(n)
    
    def create_length_distribution_plot(self):
        """Create plots showing response length distributions"""
        if self.responses_df.empty:
            return None
        
        responses = self.responses_df['response'].dropna()
        char_lengths = responses.str.len()
        word_lengths = responses.str.split().str.len()
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Character Length Distribution', 'Word Length Distribution')
        )
        
        fig.add_trace(
            go.Histogram(x=char_lengths, name='Character Length', nbinsx=30),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Histogram(x=word_lengths, name='Word Length', nbinsx=20),
            row=1, col=2
        )
        
        fig.update_layout(
            title_text="Response Length Analysis",
            showlegend=False,
            height=400
        )
        
        return fig


class CategoryAnalyzer:
    """Utility class for category analysis"""
    
    def __init__(self, categorized_responses: Dict, categories: Dict):
        self.categorized_responses = categorized_responses
        self.categories = categories
    
    def calculate_category_statistics(self) -> Dict:
        """Calculate statistics for each category"""
        if not self.categorized_responses or not self.categories:
            return {}
        
        category_stats = {}
        category_keys = list(self.categories.get('categories', {}).keys())
        
        for category_key in category_keys:
            category_name = self.categories['categories'][category_key]['name']
            count = sum(1 for response in self.categorized_responses.values() 
                       if response.get(category_key, 0) == 1)
            
            category_stats[category_name] = {
                'count': count,
                'percentage': (count / len(self.categorized_responses)) * 100 if self.categorized_responses else 0
            }
        
        return category_stats
    
    def create_category_heatmap(self):
        """Create heatmap showing category co-occurrence"""
        if not self.categorized_responses or not self.categories:
            return None
        
        category_keys = list(self.categories.get('categories', {}).keys())
        category_names = [self.categories['categories'][key]['name'] for key in category_keys]
        
        # Create co-occurrence matrix
        cooccurrence_matrix = np.zeros((len(category_keys), len(category_keys)))
        
        for response in self.categorized_responses.values():
            active_categories = [i for i, key in enumerate(category_keys) 
                               if response.get(key, 0) == 1]
            
            for i in active_categories:
                for j in active_categories:
                    cooccurrence_matrix[i][j] += 1
        
        fig = go.Figure(data=go.Heatmap(
            z=cooccurrence_matrix,
            x=category_names,
            y=category_names,
            colorscale='Blues'
        ))
        
        fig.update_layout(
            title="Category Co-occurrence Heatmap",
            xaxis_title="Categories",
            yaxis_title="Categories"
        )
        
        return fig
    
    def identify_multi_category_responses(self) -> List[Dict]:
        """Identify responses that belong to multiple categories"""
        multi_category = []
        
        for response_id, response_data in self.categorized_responses.items():
            category_count = sum(1 for key, value in response_data.items() 
                               if key.startswith('bucket') and value == 1)
            
            if category_count > 1:
                active_categories = [key for key, value in response_data.items() 
                                   if key.startswith('bucket') and value == 1]
                multi_category.append({
                    'response_id': response_id,
                    'category_count': category_count,
                    'categories': active_categories,
                    'phrases': response_data.get('identified_phrases', '')
                })
        
        return multi_category


class ReportGenerator:
    """Generate comprehensive analysis reports"""
    
    def __init__(self, responses_df: pd.DataFrame, categories: Dict, 
                 categorized_responses: Dict, top_quotes: Dict):
        self.responses_df = responses_df
        self.categories = categories
        self.categorized_responses = categorized_responses
        self.top_quotes = top_quotes
    
    def generate_executive_summary(self) -> str:
        """Generate executive summary of the analysis"""
        if self.responses_df.empty:
            return "No data available for analysis."
        
        total_responses = len(self.responses_df)
        avg_length = self.responses_df['response'].str.len().mean()
        
        # Category statistics
        analyzer = CategoryAnalyzer(self.categorized_responses, self.categories)
        category_stats = analyzer.calculate_category_statistics()
        
        top_categories = sorted(category_stats.items(), 
                              key=lambda x: x[1]['count'], reverse=True)[:3]
        
        summary = f"""
        ## Executive Summary
        
        **Total Responses Analyzed:** {total_responses}
        **Average Response Length:** {avg_length:.0f} characters
        
        **Top Categories:**
        """
        
        for i, (category, stats) in enumerate(top_categories, 1):
            summary += f"\n{i}. **{category}**: {stats['count']} responses ({stats['percentage']:.1f}%)"
        
        if self.top_quotes:
            summary += "\n\n**Key Insights:**\n"
            for category, quotes in list(self.top_quotes.items())[:2]:
                if quotes:
                    summary += f"\n- **{category}**: \"{quotes[0].get('quote', '')}\""
        
        return summary
    
    def create_comprehensive_report(self) -> str:
        """Create a comprehensive text report"""
        report = self.generate_executive_summary()
        
        # Add detailed category breakdown
        analyzer = CategoryAnalyzer(self.categorized_responses, self.categories)
        category_stats = analyzer.calculate_category_statistics()
        
        report += "\n\n## Detailed Category Analysis\n"
        
        for category, stats in category_stats.items():
            report += f"\n### {category}\n"
            report += f"- Responses: {stats['count']}\n"
            report += f"- Percentage: {stats['percentage']:.1f}%\n"
            
            # Add top quotes for this category
            if category in self.top_quotes and self.top_quotes[category]:
                report += "- Top Quote: \"" + self.top_quotes[category][0].get('quote', '') + "\"\n"
        
        return report


# ======================
# Visualization Utilities
# ======================

def create_category_distribution_plot(category_stats: Dict) -> go.Figure:
    """Create a bar plot showing distribution of responses across categories"""
    if not category_stats:
        return None
    
    categories = list(category_stats.keys())
    counts = [stats['count'] for stats in category_stats.values()]
    percentages = [stats['percentage'] for stats in category_stats.values()]
    
    fig = go.Figure(data=[
        go.Bar(
            x=categories,
            y=counts,
            text=[f"{count} ({pct:.1f}%)" for count, pct in zip(counts, percentages)],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Response Distribution Across Categories",
        xaxis_title="Categories",
        yaxis_title="Number of Responses",
        xaxis_tickangle=-45
    )
    
    return fig

def create_analysis_dashboard(responses_df: pd.DataFrame, categories: Dict, 
                            categorized_responses: Dict, top_quotes: Dict):
    """Create a comprehensive analysis dashboard"""
    st.header("📊 Analysis Dashboard")
    
    # Response statistics
    analyzer = ResponseAnalyzer(responses_df)
    stats = analyzer.get_response_statistics()
    
    if stats:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Responses", stats['total_responses'])
        
        with col2:
            st.metric("Avg Word Count", f"{stats['word_length']['mean']:.0f}")
        
        with col3:
            st.metric("Avg Character Count", f"{stats['char_length']['mean']:.0f}")
        
        # Length distribution plot
        length_plot = analyzer.create_length_distribution_plot()
        if length_plot:
            st.plotly_chart(length_plot, use_container_width=True)
    
    # Category analysis
    if categorized_responses and categories:
        category_analyzer = CategoryAnalyzer(categorized_responses, categories)
        category_stats = category_analyzer.calculate_category_statistics()
        
        # Category distribution
        dist_plot = create_category_distribution_plot(category_stats)
        if dist_plot:
            st.plotly_chart(dist_plot, use_container_width=True)
        
        # Category heatmap
        heatmap = category_analyzer.create_category_heatmap()
        if heatmap:
            st.plotly_chart(heatmap, use_container_width=True)
        
        # Multi-category responses
        multi_cat = category_analyzer.identify_multi_category_responses()
        if multi_cat:
            st.subheader("Multi-Category Responses")
            st.write(f"Found {len(multi_cat)} responses spanning multiple categories")
            
            for response in multi_cat[:5]:  # Show top 5
                st.write(f"**Response {response['response_id']}**: {response['category_count']} categories")
                st.write(f"Phrases: {response['phrases']}")


# ======================
# Helper Functions
# ======================

def load_prompt_template(template_name: str, prompts_dir: str = "prompts") -> str:
    """
    Load a prompt template from file
    
    Args:
        template_name: Name of the template file (without .md extension)
        prompts_dir: Directory containing prompt templates
        
    Returns:
        Template content as string
    """
    template_path = Path(prompts_dir) / f"{template_name}.md"
    
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        st.error(f"Prompt template not found: {template_path}")
        return ""
    except Exception as e:
        st.error(f"Error loading prompt template: {e}")
        return ""

def save_analysis_results(results: Dict, output_dir: str = "output"):
    """
    Save analysis results to files
    
    Args:
        results: Dictionary containing analysis results
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save categorized responses
    if 'categorized_responses' in results:
        with open(f"{output_dir}/categorized_responses.json", 'w') as f:
            json.dump(results['categorized_responses'], f, indent=2)
    
    # Save categories
    if 'categories' in results:
        with open(f"{output_dir}/categories.json", 'w') as f:
            json.dump(results['categories'], f, indent=2)
    
    # Save top quotes
    if 'top_quotes' in results:
        with open(f"{output_dir}/top_quotes.json", 'w') as f:
            json.dump(results['top_quotes'], f, indent=2)
    
    # Save as CSV if responses dataframe exists
    if 'responses_df' in results:
        results['responses_df'].to_csv(f"{output_dir}/responses.csv", index=False)

def validate_environment():
    """Validate that required environment variables are set"""
    required_vars = ['OPENAI_API_KEY']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        st.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        return False
    
    return True

def clean_response_text(text: str) -> str:
    """Clean and normalize response text"""
    if not text or pd.isna(text):
        return ""
    
    # Convert to string and strip whitespace
    text = str(text).strip()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters that might cause issues
    text = re.sub(r'[^\w\s.,!?;:()\-\'\"]+', '', text)
    
    return text

def format_percentage(value: float, decimals: int = 1) -> str:
    """Format a decimal as a percentage string"""
    return f"{value:.{decimals}f}%"

def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text to specified length with ellipsis"""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."
