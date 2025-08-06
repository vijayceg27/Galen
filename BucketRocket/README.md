# BucketRocket

A streamlined AI-powered tool for pharmaceutical market research that identifies categories from survey responses and extracts scored quotes. Features advanced LiteLLM integration and comprehensive scoring mechanisms for professional research analysis.

## 🚀 Key Features

- **🔍 Category Identification**: Analyze ALL survey responses to identify 5-15 thematic categories
- **📊 Advanced Quote Scoring**: Score and rank ALL responses (0-100 scale) with detailed breakdown
- **🎯 Interactive Filtering**: Filter quotes by score threshold with color-coded visual indicators
- **📈 Category Analytics**: View response counts, average scores, and top themes per category
- **🔗 Admin Portal Integration**: Export/import CSV files for seamless workflow integration
- **🤖 LiteLLM Support**: Unified interface for multiple LLM providers (OpenAI, Anthropic, Gemini, etc.)

## 📋 Prerequisites

- **Database**: MySQL/MariaDB with survey response data
- **API Access**: LiteLLM proxy or OpenAI API key
- **Python**: 3.8+ with required packages

## ⚙️ Quick Setup

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your database and API credentials
   ```

3. **Run the application**
   ```bash
   python -m streamlit run app.py
   ```

## 🎯 Advanced Scoring System

### Scoring Criteria (0-100 Scale)
- **Relevance (30 points)**: How well the response fits the category theme
- **Clarity (25 points)**: Clear, specific, and well-articulated content
- **Insight Value (25 points)**: Actionable insights for pharmaceutical research
- **Professional Perspective (20 points)**: Healthcare professional viewpoint
- **Bonus Points (15 points)**: Specific examples, unique insights, actionable recommendations

### Interactive Features
- **Score Filtering**: Adjustable minimum threshold (0-100)
- **Color Coding**: 🟢 High (≥80), 🟡 Medium (≥60), 🔴 Low (<60)
- **Detailed Analysis**: Comprehensive rationale for each score
- **Category Metrics**: Total responses, average scores, top themes

## 📋 Workflow

### Step 1: Project Selection
1. Select **Project** from dropdown
2. Select **Survey** (filtered by project)
3. Select **Wave** (filtered by survey with date ranges)
4. Choose **Question** to analyze

### Step 2: Category Identification
1. Configure number of categories (5-15)
2. Click "🔍 **Identify Categories**"
3. Review AI-generated categories with descriptions
4. Download CSV for admin portal upload

### Step 3: Admin Portal Integration
1. Upload categories CSV to admin portal
2. Admin portal categorizes all responses
3. Download categorized responses CSV

### Step 4: Quote Extraction & Scoring
1. Upload categorized CSV to analyzer
2. Select category or "All Categories"
3. Click "💬 **Extract Top Quotes**"
4. View scored responses with interactive filtering
5. Download comprehensive results as JSON

## 🛠️ Configuration

### Environment Variables
```bash
# Database
DB_HOST=your_database_host
DB_USER=your_database_user
DB_PASSWORD=your_database_password
DB_NAME=your_database_name

# LiteLLM Proxy
LLM_API_KEY=your_api_key
LLM_API_BASE=https://your-litellm-proxy-url
LLM_MODEL_NAME=openai/gpt-4o-mini

# Optional
LITELLM_VERBOSE=False
LITELLM_RETRIES=3
```

## 📁 Project Structure

```
BucketRocket/
├── app.py                  # Main Streamlit application
├── llm_service.py          # LLM integration and scoring logic
├── utils.py                # Database utilities
├── requirements.txt        # Python dependencies
├── .env.template          # Environment template
└── prompts/               # AI prompt templates
    ├── identify_categories.md
    └── extract_quotes.md
```

## 🔧 Technical Details

### Database Schema
- **Tables**: `lime_questions`, `survey_responses`, `users_waves`, `waves`, `projects`, `projects_surveys`
- **Key Columns**: JSON `responses` column in `survey_responses`

### LLM Integration
- **LiteLLM Proxy**: Unified API for multiple providers
- **Fallback Support**: Direct OpenAI integration if needed
- **Retry Logic**: Automatic retry with exponential backoff

### Data Flow
1. **Database Query**: Fetch responses from specific wave
2. **Category Analysis**: LLM identifies thematic categories
3. **CSV Export**: Admin portal compatible format
4. **Quote Scoring**: Comprehensive 0-100 scale evaluation
5. **Interactive Display**: Filtered, scored results with analysis

## 🎯 Benefits

- **Comprehensive Analysis**: ALL responses analyzed, not just samples
- **Objective Scoring**: Consistent criteria across all responses
- **Quality Control**: Filter responses by relevance scores
- **Professional Focus**: Tailored for pharmaceutical market research
- **Seamless Integration**: Works with existing admin portal workflow
- **Scalable**: Handles large volumes of responses efficiently

## 📞 Support

For issues or questions:
1. Check environment variable configuration
2. Verify database connectivity
3. Confirm API key validity
4. Review Streamlit error messages

---

**Built for pharmaceutical market research professionals who need AI-driven insights from qualitative survey data.**
