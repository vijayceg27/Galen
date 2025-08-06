"""
Enhanced Text Response Analyzer for Pharmaceutical Market Research
Comprehensive tool for analyzing survey responses with LLM-powered categorization
"""

import streamlit as st
import shutil
import os
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from utils import (connect_db, fetch_projects, fetch_surveys, fetch_waves, fetch_questions, fetch_responses,
                   ResponseAnalyzer, CategoryAnalyzer, ReportGenerator, create_analysis_dashboard)
from llm_service import LLMService

# Page configuration
st.set_page_config(
    page_title="Text Response Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #f0f8ff, #e6f3ff);
        border-radius: 10px;
        border: 2px solid #1f77b4;
    }
    
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin: 1.5rem 0 1rem 0;
        padding: 0.5rem;
        background-color: #f8f9fa;
        border-left: 4px solid #3498db;
        border-radius: 5px;
    }
    
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
    }
    
    .category-box {
        background-color: #fff;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .stExpander {
        border: 1px solid #dee2e6;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'projects_df' not in st.session_state:
        st.session_state.projects_df = pd.DataFrame()
    if 'surveys_df' not in st.session_state:
        st.session_state.surveys_df = pd.DataFrame()
    if 'waves_df' not in st.session_state:
        st.session_state.waves_df = pd.DataFrame()
    if 'selected_project_id' not in st.session_state:
        st.session_state.selected_project_id = None
    if 'selected_survey_id' not in st.session_state:
        st.session_state.selected_survey_id = None
    if 'selected_wave_id' not in st.session_state:
        st.session_state.selected_wave_id = None
    if 'last_project_id' not in st.session_state:
        st.session_state.last_project_id = None
    if 'last_survey_id' not in st.session_state:
        st.session_state.last_survey_id = None
    if 'responses_df' not in st.session_state:
        st.session_state.responses_df = pd.DataFrame()
    if 'identified_categories' not in st.session_state:
        st.session_state.identified_categories = None
    if 'admin_portal_data' not in st.session_state:
        st.session_state.admin_portal_data = None
    if 'extracted_quotes' not in st.session_state:
        st.session_state.extracted_quotes = None

initialize_session_state()

def create_project_structure(project_id: str, survey_id: str) -> str:
    """Create project folder structure and copy environment file"""
    try:
        # Create project directory structure
        project_dir = f"projects/project_{project_id}/survey_{survey_id}"
        os.makedirs(project_dir, exist_ok=True)
        
        # Copy .env file to project directory
        env_source = ".env"
        env_destination = os.path.join(project_dir, ".env")
        
        if os.path.exists(env_source) and not os.path.exists(env_destination):
            shutil.copy2(env_source, env_destination)
        
        return env_destination
    except Exception as e:
        st.warning(f"Could not create project structure: {e}")
        return ".env"

def display_response_overview(df: pd.DataFrame):
    """Display overview statistics of responses"""
    st.markdown('<div class="section-header">📊 Response Overview</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Responses", len(df))
    
    with col2:
        avg_length = df['response'].str.len().mean()
        st.metric("Avg Length (chars)", f"{avg_length:.0f}")
    
    with col3:
        avg_words = df['response'].str.split().str.len().mean()
        st.metric("Avg Words", f"{avg_words:.0f}")
    
    with col4:
        valid_responses = df['response'].notna().sum()
        st.metric("Valid Responses", valid_responses)

def display_categories(categories: Dict):
    """
    Display identified categories in a structured format
    
    Args:
        categories: Dictionary with categories from identify_categories
    """
    st.markdown("#### 📋 Identified Categories")
    
    for i, (category_name, category_info) in enumerate(categories.items(), 1):
        with st.expander(f"{i}. {category_name}", expanded=False):
            if isinstance(category_info, dict):
                if 'description' in category_info:
                    st.markdown(f"**Description:** {category_info['description']}")
                if 'examples' in category_info:
                    st.markdown("**Example responses:**")
                    for example in category_info['examples']:
                        st.markdown(f"• {example}")
                if 'keywords' in category_info:
                    st.markdown(f"**Keywords:** {', '.join(category_info['keywords'])}")
            else:
                st.markdown(f"**Description:** {category_info}")

def main():
    """Main application function"""
    
    # Header
    st.markdown('<div class="main-header">🔬 Pharma Survey Text Analyzer</div>', unsafe_allow_html=True)
    st.markdown("### Comprehensive tool for analyzing survey responses with LLM-powered categorization")
    
    # Sidebar for configuration and navigation
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        # LLM Provider Selection
        llm_provider = st.radio(
            "🤖 LLM Provider",
            options=["LiteLLM Proxy", "OpenAI Direct"],
            index=0,
            help="Choose between LiteLLM proxy or direct OpenAI API"
        )
        
        if llm_provider == "LiteLLM Proxy":
            st.subheader("🔌 LiteLLM Proxy Settings")
            llm_api_key = st.text_input(
                "LiteLLM API Key",
                type="password",
                help="Your LiteLLM proxy API key"
            )
            llm_api_base = st.text_input(
                "LiteLLM API Base URL",
                value="https://llm-local.zoomrx.dev",
                help="Base URL for your LiteLLM proxy"
            )
            llm_model = st.selectbox(
                "Model",
                options=[
                    "openai/o4-mini",
                    "openai/gpt-4.1-mini",
                    "openai/gpt-4.1-nano",
                    "openai/gpt-4o-mini",
                    "gemini/gemini-2.5-flash-lite",
                    "gemini/gemini-2.5-flash"
                ],
                index=0,
                help="Select the model to use for analysis"
            )
            
            if not llm_api_key or not llm_api_base:
                st.warning("⚠️ Please provide LiteLLM API key and base URL")
                st.stop()
            
            # Set environment variables for LiteLLM
            os.environ["LLM_API_KEY"] = llm_api_key
            os.environ["LLM_API_BASE"] = llm_api_base
            os.environ["LLM_MODEL_NAME"] = llm_model
            
        else:
            st.subheader("🔑 OpenAI Settings")
            api_key = st.text_input(
                "OpenAI API Key",
                type="password",
                help="Your OpenAI API key"
            )
            
            if not api_key:
                st.warning("⚠️ Please provide your OpenAI API key")
                st.stop()
            
            os.environ["OPENAI_API_KEY"] = api_key
        
        st.session_state.api_key = api_key if llm_provider == "OpenAI Direct" else llm_api_key
        
        st.markdown("---")
        
        # Database connection and project selection
        try:
            conn = connect_db()
            
            # Project dropdown
            if st.session_state.projects_df.empty:
                st.session_state.projects_df = fetch_projects(conn)
            
            if not st.session_state.projects_df.empty:
                project_options = {}
                for _, row in st.session_state.projects_df.iterrows():
                    project_display = f"{row['name']} (ID: {row['id']})"
                    project_options[row['id']] = project_display
                
                project_display_list = list(project_options.values())
                
                selected_project_display = st.selectbox(
                    "🏢 Select Project",
                    options=project_display_list,
                    help="Choose a project from the database"
                )
                
                if selected_project_display:
                    # Extract project ID from display string
                    project_id = [k for k, v in project_options.items() if v == selected_project_display][0]
                    st.session_state.selected_project_id = project_id
                    
                    # Survey dropdown
                    if project_id != st.session_state.get('last_project_id'):
                        st.session_state.surveys_df = fetch_surveys(conn, project_id)
                        st.session_state.last_project_id = project_id
                        st.session_state.selected_survey_id = None
                    
                    if not st.session_state.surveys_df.empty:
                        survey_options = {}
                        for _, row in st.session_state.surveys_df.iterrows():
                            survey_display = f"{row['surveyls_title']} (ID: {row['survey_id']})"
                            survey_options[row['survey_id']] = survey_display
                        
                        survey_display_list = list(survey_options.values())
                        
                        selected_survey_display = st.selectbox(
                            "📋 Select Survey",
                            options=survey_display_list,
                            help="Choose a survey from the selected project"
                        )
                        
                        if selected_survey_display:
                            # Extract survey ID from display string
                            survey_id = [k for k, v in survey_options.items() if v == selected_survey_display][0]
                            st.session_state.selected_survey_id = survey_id
                            
                            # Wave dropdown
                            if survey_id != st.session_state.get('last_survey_id'):
                                st.session_state.waves_df = fetch_waves(conn, survey_id)
                                st.session_state.last_survey_id = survey_id
                                st.session_state.selected_wave_id = None
                            
                            if not st.session_state.waves_df.empty:
                                wave_options = {}
                                for _, row in st.session_state.waves_df.iterrows():
                                    wave_display = f"Wave {row['id']}: {row['start_date']} to {row['end_date']}"
                                    wave_options[row['id']] = wave_display
                                
                                wave_display_list = list(wave_options.values())
                                
                                selected_wave_display = st.selectbox(
                                    "🌊 Select Wave",
                                    options=wave_display_list,
                                    help="Choose a wave from the selected survey"
                                )
                                
                                if selected_wave_display:
                                    # Extract wave ID from display string
                                    wave_id = [k for k, v in wave_options.items() if v == selected_wave_display][0]
                                    st.session_state.selected_wave_id = wave_id
                                else:
                                    st.info("👆 Please select a wave to continue")
                                    st.stop()
                            else:
                                st.warning("No waves found for this survey")
                                st.stop()
                        else:
                            st.info("👆 Please select a survey to continue")
                            st.stop()
                    else:
                        st.warning("No surveys found for this project")
                        st.stop()
                else:
                    st.info("👆 Please select a project to continue")
                    st.stop()
            else:
                st.error("No projects found in database")
                st.stop()
                
        except Exception as e:
            st.error(f"Database connection error: {e}")
            st.stop()
    
    # Main content
    try:
        # Use selected values from dropdowns
        project_id = st.session_state.selected_project_id
        survey_id = st.session_state.selected_survey_id
        wave_id = st.session_state.selected_wave_id
        
        # Create project structure
        env_path = create_project_structure(str(project_id), str(survey_id))
        
        # Fetch questions
        with st.spinner("📋 Fetching survey questions..."):
            questions = fetch_questions(conn, survey_id)
        
        if not questions:
            st.error("❌ No open-ended questions found for this survey ID")
            st.stop()
        
        # Question selection
        st.markdown('<div class="section-header">📝 Select Question</div>', unsafe_allow_html=True)
        
        question_options = {f"{q['QCode']} - {q['question'][:100]}...": q for q in questions}
        selected_question_key = st.selectbox(
            "Choose a question to analyze:",
            list(question_options.keys()),
            help="Select the open-ended question you want to analyze"
        )
        
        selected_question = question_options[selected_question_key]
        question_id = selected_question['qid']
        question_text = selected_question['question']
        
        st.session_state.selected_question = selected_question
        
        # Display question details
        st.info(f"**Question Code:** {selected_question['QCode']}")
        st.info(f"**Question:** {question_text}")
        
        # Fetch responses
        with st.spinner(f"📊 Fetching responses for Wave {wave_id}..."):
            responses_df = fetch_responses(conn, str(survey_id), question_id, wave_id)
            st.session_state.responses_df = responses_df
        
        if responses_df.empty:
            st.warning("⚠️ No responses found for this question")
            st.stop()
        
        # Display response overview
        display_response_overview(responses_df)
        
        # Show sample responses with controls
        with st.expander(f"👀 View Sample Responses (Total: {len(responses_df)} responses)"):
            col1, col2 = st.columns([3, 1])
            with col2:
                sample_size = st.selectbox(
                    "Sample size:", 
                    options=[10, 25, 50, 100, len(responses_df)],
                    index=0,
                    key="sample_size_display"
                )
            
            if sample_size == len(responses_df):
                st.info(f"Showing all {len(responses_df)} responses")
                sample_df = responses_df
            else:
                st.info(f"Showing {min(sample_size, len(responses_df))} of {len(responses_df)} responses")
                sample_df = responses_df.head(sample_size)
            
            st.dataframe(sample_df, use_container_width=True)
        
        # Analysis section
        st.markdown('<div class="section-header">🤖 AI Analysis</div>', unsafe_allow_html=True)
        
        # Analysis configuration
        with st.expander("⚙️ Analysis Configuration", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                num_categories = st.selectbox(
                    "Number of categories to identify:",
                    options=[5, 8, 10, 12, 15],
                    index=2,  # Default to 10
                    help="Number of high-level categories to identify from responses"
                )
            with col2:
                st.info(f"Total responses to analyze: {len(responses_df)}")
        
        # Step 1: Category Identification
        st.markdown("### Step 1: Identify Categories")
        
        # Initialize LLM service
        try:
            llm_service = LLMService()
        except Exception as e:
            st.error(f"Failed to initialize LLM service: {str(e)}")
            st.stop()
        
        # Category identification button
        if st.button("🔍 Identify Categories", type="primary", use_container_width=True):
            with st.spinner(f"Analyzing {len(responses_df)} responses to identify {num_categories} categories..."):
                try:
                    # Get responses as list
                    responses_list = responses_df['response'].tolist()
                    
                    # Identify categories using LLM
                    categories = llm_service.identify_categories(
                        responses=responses_list,
                        question_text=question_text,
                        num_categories=num_categories
                    )
                    
                    if categories:
                        st.session_state.identified_categories = categories
                        st.success(f"✅ Successfully identified {len(categories)} categories!")
                        
                        # Display categories
                        display_categories(categories)
                        
                        # Export categories to CSV for admin portal
                        csv_content = llm_service.export_categories_csv(categories, include_subcategories=True)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.download_button(
                                label="📥 Download Categories CSV (for Admin Portal)",
                                data=csv_content,
                                file_name=f"categories_project_{project_id}_survey_{survey_id}_question_{question_id}.csv",
                                mime="text/csv",
                                help="Download this CSV file and upload it to the admin portal for categorization"
                            )
                        
                        with col2:
                            # Also provide JSON backup
                            categories_json = json.dumps(categories, indent=2)
                            st.download_button(
                                label="📄 Download Categories JSON (backup)",
                                data=categories_json,
                                file_name=f"categories_project_{project_id}_survey_{survey_id}_question_{question_id}.json",
                                mime="application/json"
                            )
                        
                        st.info("💡 **Next Steps:** Download the CSV file above and upload it to the admin portal for response categorization. Then return here and upload the categorized results in Step 2.")
                    else:
                        st.error("Failed to identify categories. Please try again.")
                        
                except Exception as e:
                    st.error(f"Error during category identification: {str(e)}")
                    st.exception(e)
        
        # Show identified categories if they exist
        if 'identified_categories' in st.session_state and st.session_state.identified_categories:
            st.markdown("#### ✅ Identified Categories")
            display_categories(st.session_state.identified_categories)
        
        # Step 2: Upload Categorized Results
        st.markdown("### Step 2: Upload Categorized Results from Admin Portal")
        
        # CSV file upload for admin portal results
        uploaded_file = st.file_uploader(
            "📤 Upload categorized results CSV from admin portal",
            type=['csv'],
            help="Upload the CSV file you downloaded from the admin portal after categorization"
        )
        
        if uploaded_file is not None:
            try:
                # Save uploaded file temporarily
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                # Process the uploaded CSV
                with st.spinner("Processing uploaded CSV file..."):
                    admin_portal_data = llm_service.process_admin_portal_csv(tmp_file_path)
                    st.session_state.admin_portal_data = admin_portal_data
                
                st.success("✅ CSV file processed successfully!")
                
                # Display processing results
                category_columns = admin_portal_data.get('available_categories', [])
                total_responses = admin_portal_data.get('total_responses', 0)
                
                st.info(f"📊 Found {len(category_columns)} categories and {total_responses} total responses")
                
                if category_columns:
                    st.markdown("#### Available Categories:")
                    cols = st.columns(min(3, len(category_columns)))
                    for i, cat in enumerate(category_columns[:9]):  # Show first 9 categories
                        with cols[i % 3]:
                            # Count responses in this category
                            count = len(admin_portal_data['categorized_responses'].get(cat, []))
                            st.metric(cat, f"{count} responses")
                    
                    if len(category_columns) > 9:
                        st.info(f"... and {len(category_columns) - 9} more categories")
                
                # Clean up temporary file
                try:
                    os.unlink(tmp_file_path)
                except Exception:
                    pass
                    
            except Exception as e:
                st.error(f"Error processing CSV file: {str(e)}")
                # Clean up temporary file on error
                if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
                    try:
                        os.unlink(tmp_file_path)
                    except Exception:
                        pass
        
        # Step 3: Quote Extraction (only show if admin portal data exists)
        if 'admin_portal_data' in st.session_state and st.session_state.admin_portal_data:
            st.markdown("### Step 3: Extract Top Quotes")
            admin_portal_data = st.session_state.admin_portal_data
            category_columns = admin_portal_data.get('available_categories', [])
            
            if category_columns:
                # Category selection for quote extraction
                selected_category = st.selectbox(
                    "Select category for quote extraction:",
                    options=['All Categories'] + category_columns,
                    help="Choose a specific category or extract quotes for all categories",
                    key="selected_category"
                )
                
                # Extract quotes button
                if st.button("💬 Extract Top Quotes", type="primary", use_container_width=True):
                    with st.spinner(f"Extracting compelling quotes for {selected_category}..."):
                        category_to_process = None if selected_category == 'All Categories' else selected_category
                        
                        quotes = llm_service.extract_top_quotes_from_csv(
                            admin_portal_data, 
                            selected_category=category_to_process
                        )
                        
                        if quotes:
                            st.session_state.extracted_quotes = quotes
                            st.success("✅ Quotes extracted successfully!")
                            
                            # Display quotes with scoring
                            for category_name, category_data in quotes.items():
                                if category_data and 'scored_quotes' in category_data:
                                    scored_quotes = category_data['scored_quotes']
                                    category_summary = category_data.get('category_summary', {})
                                    
                                    st.markdown(f"#### 💬 {category_name}")
                                    
                                    # Display category summary
                                    if category_summary:
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            st.metric("Total Responses", category_summary.get('total_responses', 0))
                                        with col2:
                                            avg_score = category_summary.get('average_score', 0)
                                            st.metric("Average Score", f"{avg_score:.1f}/100")
                                        with col3:
                                            top_themes = category_summary.get('top_themes', [])
                                            if top_themes:
                                                st.metric("Top Themes", len(top_themes))
                                    
                                    # Display scored quotes
                                    if scored_quotes:
                                        st.markdown("##### 📊 Scored Responses (Ranked by Relevance)")
                                        
                                        # Add score filter
                                        min_score = st.slider(
                                            f"Minimum score for {category_name}",
                                            min_value=0,
                                            max_value=100,
                                            value=60,
                                            step=5,
                                            key=f"min_score_{category_name}"
                                        )
                                        
                                        # Filter quotes by minimum score
                                        filtered_quotes = [q for q in scored_quotes if q.get('total_score', 0) >= min_score]
                                        
                                        if filtered_quotes:
                                            st.info(f"Showing {len(filtered_quotes)} of {len(scored_quotes)} responses (score ≥ {min_score})")
                                            
                                            for i, quote in enumerate(filtered_quotes, 1):
                                                total_score = quote.get('total_score', 0)
                                                score_breakdown = quote.get('score_breakdown', {})
                                                bonus_points = quote.get('bonus_points', 0)
                                                
                                                # Color code based on score
                                                if total_score >= 80:
                                                    score_color = "🟢"  # Green for high scores
                                                elif total_score >= 60:
                                                    score_color = "🟡"  # Yellow for medium scores
                                                else:
                                                    score_color = "🔴"  # Red for low scores
                                                
                                                with st.expander(f"{score_color} Response {i}: Score {total_score}/100", expanded=i <= 3):
                                                    # Display the quote
                                                    st.markdown(f"**Response:** *\"{quote.get('quote', 'N/A')}\"*")
                                                    
                                                    # Display score breakdown
                                                    st.markdown("**Score Breakdown:**")
                                                    col1, col2 = st.columns(2)
                                                    
                                                    with col1:
                                                        st.markdown(f"• Relevance: {score_breakdown.get('relevance', 0)}/30")
                                                        st.markdown(f"• Clarity: {score_breakdown.get('clarity', 0)}/25")
                                                    
                                                    with col2:
                                                        st.markdown(f"• Insight Value: {score_breakdown.get('insight_value', 0)}/25")
                                                        st.markdown(f"• Professional Perspective: {score_breakdown.get('professional_perspective', 0)}/20")
                                                    
                                                    if bonus_points > 0:
                                                        st.markdown(f"• **Bonus Points:** +{bonus_points}")
                                                    
                                                    # Display rationale
                                                    rationale = quote.get('rationale', 'No rationale provided')
                                                    st.markdown(f"**Analysis:** {rationale}")
                                                    
                                                    # Response ID for reference
                                                    response_id = quote.get('response_id', 'N/A')
                                                    st.caption(f"Response ID: {response_id}")
                                        else:
                                            st.warning(f"No responses meet the minimum score of {min_score} for {category_name}")
                                    else:
                                        st.warning(f"No scored quotes available for {category_name}")
                                elif category_data:
                                    # Handle legacy format (backward compatibility)
                                    st.markdown(f"#### 💬 {category_name}")
                                    for i, quote in enumerate(category_data, 1):
                                        with st.expander(f"Quote {i}: {quote.get('reason', 'Compelling quote')}"):
                                            st.markdown(f"**Quote:** *\"{quote.get('quote', 'N/A')}\"*")
                                            st.markdown(f"**Why it's compelling:** {quote.get('reason', 'N/A')}")
                            
                            # Export quotes
                            quotes_json = json.dumps(quotes, indent=2)
                            st.download_button(
                                label="💾 Download Extracted Quotes (JSON)",
                                data=quotes_json,
                                file_name=f"quotes_project_{project_id}_survey_{survey_id}_question_{question_id}.json",
                                mime="application/json"
                            )
                        else:
                            st.warning("No quotes could be extracted. Please check the categorized data.")
            else:
                st.warning("No category columns found in the uploaded CSV.")
        
        # Close database connection if it exists
        if 'conn' in locals():
            try:
                conn.close()
            except Exception as close_error:
                st.error(f"Error closing database connection: {close_error}")
    
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        st.exception(e)

if __name__ == "__main__":
    main()
