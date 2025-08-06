# Text Response Analyzer

A streamlined AI-powered tool for identifying comprehensive categories from pharmaceutical market research survey responses. This application uses advanced LLM technology to analyze ALL responses and generate category frameworks that can be uploaded to your production admin portal for automatic categorization.

## 🚀 Features

### Core Functionality
- **Project Management**: Organize analysis by Project ID and Survey ID
- **Question Selection**: Automatic detection of open-ended (text) questions from surveys
- **Response Retrieval**: Fetch ALL verbatim responses from JSON database columns
- **Comprehensive Category Identification**: Analyze ALL responses to identify themes (no sampling)
- **Configurable Categories**: Choose number of categories to identify (5, 8, 10, 12, 15)
- **Admin Portal Integration**: Export categories for upload to production categorization system
- **Advanced Quote Extraction**: Extract and score ALL quotes with comprehensive relevance ranking
- **Scoring Mechanism**: 0-100 point scoring system based on relevance, clarity, insight value, and professional perspective
- **Interactive Filtering**: Filter quotes by minimum score threshold with visual indicators
- **LiteLLM Integration**: Unified interface for multiple LLM providers through LiteLLM proxy

### AI Capabilities
- **Comprehensive Analysis**: Analyzes ALL responses for maximum category coverage
- **Configurable Output**: Generate 5-15 categories based on your needs
- **Intelligent Categorization**: Uses advanced prompts optimized for pharmaceutical research
- **Export Ready**: Categories formatted for direct upload to admin portal
- **Model Agnostic**: Works with multiple LLM providers (OpenAI, Anthropic, Gemini, etc.)
- **Standardized API**: Consistent interface regardless of the underlying LLM provider
- **Production Integration**: Seamless workflow with existing categorization infrastructure

### Pharmaceutical Focus
- **Domain Expertise**: Categories tailored for pharmaceutical market research
- **MSL Messaging**: Focus on medical science liaison and field force messaging themes
- **HCP Insights**: Analyze healthcare professional feedback and perspectives
- **Therapeutic Areas**: Support analysis across different therapeutic areas

## 📋 Prerequisites

### Database Requirements
- MySQL/MariaDB database with survey response data
- Tables: `lime_questions`, `survey_responses`, `users_waves`, `waves`
- JSON column `responses` in `survey_responses` table

### API Requirements
- OpenAI API key for LLM functionality
- Internet connection for API calls

### Python Environment
- Python 3.8 or higher
- Required packages (see requirements.txt)

## 📁 Project Structure

```
Text Response Analyzer/
├── enhanced_app.py          # Main Streamlit application
├── llm_service.py           # OpenAI API integration and LLM operations
├── utils.py                 # Consolidated utilities (database, analysis, visualization)
├── requirements.txt         # Python dependencies
├── .env.template           # Environment variables template
├── README.md               # This file
├── test_consolidated_utils.py # Test script for utilities
└── prompts/                # AI prompt templates
    ├── identify_categories.md
    ├── categorize_responses.md
    └── extract_quotes.md
```

### Key Components
- **`enhanced_app.py`**: Main Streamlit interface with project management and analysis workflow
- **`llm_service.py`**: Core LLM functionality with prompt management and OpenAI API integration
- **`utils.py`**: Consolidated utility functions including:
  - Database connection and data retrieval
  - Response analysis and statistics
  - Category analysis and visualization
  - Report generation
  - Helper functions for data processing
- **`prompts/`**: External prompt templates for easy modification and testing

## ⚙️ Setup

1. **Clone the repository**
   ```bash
   git clone [repository-url]
   cd text-response-analyzer
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   - Copy `.env.example` to `.env`
   - Update the following variables for database and LiteLLM:
     ```
     # Database Configuration
     DB_HOST=your_database_host
     DB_USER=your_database_user
     DB_PASSWORD=your_database_password
     DB_NAME=your_database_name
     
     # LiteLLM Configuration
     LLM_API_KEY=your_llm_api_key
     LLM_API_BASE=https://your-litellm-proxy-url
     LLM_MODEL_NAME=lite_llm_alias/model-name  # e.g., lite_llm_alias/gpt-4o-mini-2024-07-18
     
     # Optional: Advanced LiteLLM settings
     LITELLM_VERBOSE=False  # Set to True for debugging
     LITELLM_TIMEOUT=60     # Request timeout in seconds
     LITELLM_RETRIES=3      # Number of retry attempts
     ```

## 🚀 Usage

### Running the Application

```bash
streamlit run app.py
```

## 📋 Streamlined Workflow

### Step 1: Project Setup
1. **Select Project** from searchable dropdown (loads all available projects)
2. **Select Survey** from dropdown (filtered by chosen project)
3. **Select Wave** from dropdown (filtered by chosen survey with date ranges)
4. Select the specific **question** you want to analyze
5. Review the fetched responses from the selected wave (configurable sample display)

### Step 2: Category Identification
1. Choose **number of categories** to identify (5, 8, 10, 12, or 15)
2. Click "**Identify Categories**" to analyze ALL responses comprehensively
3. AI analyzes ALL response patterns (no sampling) for maximum coverage
4. Review the generated categories with detailed descriptions
5. **Download categories as CSV** in admin portal format (parent_bucket_name, bucket_name, is_active)
6. **Optional:** Download JSON backup for reference

### Step 3: Admin Portal Integration
1. Upload the downloaded **CSV file** to your production admin portal
2. Admin portal automatically categorizes all responses using the uploaded categories
3. **Download the categorized responses CSV** from admin portal (format: users_waves, question_id, voice_state, responses, responses_original, is_local_outlier, [category columns...])
4. Responses are now categorized and ready for quote extraction

### Step 4: Advanced Quote Extraction with Scoring (Optional)
1. **Upload the categorized responses CSV** from admin portal to this tool
2. Select specific category or choose "All Categories" for quote extraction
3. Click "**Extract Top Quotes**" - AI scores and ranks ALL quotes using comprehensive criteria
4. **Interactive Analysis**:
   - View category summary metrics (total responses, average score, top themes)
   - Use score filter slider to set minimum threshold (0-100)
   - See color-coded responses: 🟢 High (≥80), 🟡 Medium (≥60), 🔴 Low (<60)
   - Expand top responses for detailed breakdown
5. **Detailed Scoring Breakdown** for each response:
   - **Relevance** (0-30 points): How well it fits the category theme
   - **Clarity** (0-25 points): Clear, specific, well-articulated content
   - **Insight Value** (0-25 points): Actionable insights for pharma research
   - **Professional Perspective** (0-20 points): Healthcare professional viewpoint
   - **Bonus Points**: Specific examples, unique insights, actionable recommendations
   - **Analysis Rationale**: LLM explanation for the scoring
6. **Download comprehensive results** as JSON with all scores and analysis

### Benefits of New Workflow
- **Comprehensive**: Analyzes ALL responses, not just samples
- **Configurable**: Choose optimal number of categories for your needs
- **Integrated**: Seamless workflow with existing admin portal
- **Efficient**: Eliminates redundant categorization functionality
- **Focused**: Tool specializes in category identification and quote extraction

### Prompt Management System
The application uses a sophisticated prompt management system with templates stored in the `prompts/` directory:
- `identify_categories.md` - Identifies main themes from ALL responses
- `extract_quotes.md` - Extracts compelling quotes for each category

Features:
- Centralized prompt management
- Easy to update without code changes
- Consistent formatting and structure
- Built-in error handling for missing prompts

### 🎯 Advanced Scoring Mechanism

The application features a sophisticated scoring system that evaluates ALL responses (not just top 3) using multiple criteria:

#### Scoring Criteria (0-100 Scale)
- **Relevance (0-30 points)**: How well the response fits the category theme
- **Clarity (0-25 points)**: Clear, specific, and well-articulated content
- **Insight Value (0-25 points)**: Actionable insights for pharmaceutical research
- **Professional Perspective (0-20 points)**: Healthcare professional viewpoint
- **Bonus Points (0-15 points)**: Additional scoring for:
  - Specific examples or details (+5 points)
  - Unique or unexpected insights (+5 points)
  - Actionable recommendations (+5 points)

#### Interactive Features
- **Score Filtering**: Adjustable minimum score threshold (0-100)
- **Color Coding**: Visual indicators for score ranges
  - 🟢 **High Quality** (≥80 points): Exceptional responses with strong relevance and insights
  - 🟡 **Medium Quality** (≥60 points): Good responses with moderate relevance
  - 🔴 **Low Quality** (<60 points): Responses needing review or context
- **Expandable Details**: Top 3 responses expanded by default, others collapsible
- **Comprehensive Analysis**: Each response includes detailed rationale for scoring

#### Benefits
- **Objective Assessment**: Consistent scoring criteria across all responses
- **Quality Control**: Filter out low-relevance responses using score thresholds
- **Research Value**: Focus on responses with highest insight potential
- **Transparency**: Clear explanation of why each response received its score
- **Scalability**: Efficiently handle large volumes of responses

### Analysis Capabilities
- **Automated category identification** with comprehensive theme analysis
- **Advanced quote scoring system** (0-100 scale) with detailed breakdown
- **Interactive filtering and visualization** with color-coded responses
- **Comprehensive response analysis**:
  - Relevance scoring for category fit
  - Clarity assessment for communication quality
  - Insight value evaluation for research utility
  - Professional perspective scoring for HCP viewpoints
- **Category summary metrics** (total responses, average scores, top themes)
- **Flexible threshold filtering** for quality control
- **Detailed rationale and analysis** for each scored response
- **Export capabilities** for comprehensive reporting

## 🛠️ Testing the Application

### Prerequisites
1. Python 3.8+
2. OpenAI API key
3. Required Python packages (install via `pip install -r requirements.txt`)

### Quick Start
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```

### Running Tests
1. **Unit Tests**
   ```bash
   python -m pytest tests/
   ```

2. **Interactive Testing**
   ```python
   from llm_service import LLMService
   
   # Initialize the service
   service = LLMService()
   
   # Test with sample data
   responses = ["Great efficacy data", "Some side effects noted"]
   question = "What are your thoughts on this medication?"
   
   # Identify categories
   categories = service.identify_categories(responses, question)
   print("Identified categories:", categories)
   
   # Categorize responses
   responses_with_ids = [{"id": i, "response": r} for i, r in enumerate(responses)]
   categorized = service.categorize_responses_batch(responses_with_ids, categories, question)
   print("\nCategorized responses:", categorized)
   
   # Extract top quotes
   import pandas as pd
   df = pd.DataFrame(responses_with_ids)
   quotes = service.extract_top_quotes(categorized, categories, df)
   print("\nTop quotes:", quotes)
   ```

3. **Streamlit UI**
   ```bash
   streamlit run enhanced_app.py
   ```
   Then open your browser to `http://localhost:8501`

## 📊 Output Formats

### Categorized Responses (JSON)
```json
{
  "0": {
    "identified_phrases": "efficacy, data, effectiveness",
    "bucket1": 1,
    "bucket2": 0
  },
  "1": {
    "identified_phrases": "side effects, noted",
    "bucket1": 0,
    "bucket2": 1
  }
}
```

### Categories (JSON)
```json
{
  "categories": {
    "bucket1": {
      "name": "Efficacy/Effectiveness",
      "description": "Comments about drug effectiveness and clinical outcomes"
    },
    "bucket2": {
      "name": "Safety/Tolerability",
      "description": "Comments about side effects and safety profile"
    }
  },
  "analysis_summary": "Responses focused on efficacy and safety aspects"
}
```

### Top Quotes (JSON)
```json
{
  "Efficacy/Effectiveness": [
    {
      "id": "0",
      "quote": "Great efficacy data",
      "rationale": "Directly mentions efficacy data"
    }
  ]
}
```

### Top Quotes JSON
```json
{
  "Efficacy/Effectiveness": [
    {
      "id": "123",
      "quote": "Excellent clinical outcomes in our patient population",
      "rationale": "Clear statement of effectiveness with specific context"
    }
  ]
}
```

## 🔧 Configuration

### Environment Variables (.env)
```bash
# Database Configuration
DB_HOST=your_database_host
DB_USER=your_database_user
DB_PASSWORD=your_database_password
DB_NAME=your_database_name

# OpenAI Configuration (optional)
OPENAI_API_KEY=your_openai_api_key
```

### Database Schema Requirements

#### lime_questions table
- `qid`: Question ID
- `sid`: Survey ID  
- `title`: Question code
- `question`: Question text
- `type`: Question type ('T' for text)
- `parent_qid`: Parent question ID (0 for main questions)

#### survey_responses table
- `id`: Response ID
- `responses`: JSON column with question responses

#### users_waves table
- `id`: User ID
- `wave_id`: Wave ID

#### waves table
- `id`: Wave ID
- `survey_id`: Survey ID

## 🎯 Pharmaceutical Categories

The application identifies categories commonly relevant to pharmaceutical market research:

1. **Efficacy/Effectiveness** - Drug effectiveness and clinical outcomes
2. **Safety/Tolerability** - Side effects and safety profile
3. **Indication/Usage** - Appropriate patient populations and usage
4. **Access/Availability** - Drug access and distribution
5. **Patient Support** - Patient assistance programs
6. **Dosing/Administration** - Dosing regimens and administration
7. **Clinical Data** - Clinical trial data and evidence
8. **Competitive Positioning** - Comparisons to other treatments
9. **Cost/Economics** - Pricing and cost-effectiveness
10. **Others** - Additional themes not captured above

## 🔍 Advanced Analysis Features

### Response Statistics
- Total response count
- Average response length (characters and words)
- Response quality metrics
- Length distribution analysis

### Category Analysis
- Category distribution visualization
- Co-occurrence analysis
- Multi-category response identification
- Category correlation heatmaps

### Phrase Analysis
- Common phrase extraction
- Frequency analysis
- N-gram identification
- Keyword trend analysis

## 📈 Performance Optimization

### Batch Processing
- Responses processed in configurable batches (default: 20)
- Progress tracking for large datasets
- Error handling and retry logic

### API Efficiency
- Optimized prompts for token efficiency
- Retry logic with exponential backoff
- Cost-effective model selection (GPT-4o-mini)

### Memory Management
- Session state management
- Efficient data structures
- Lazy loading of large datasets

## 🛡️ Error Handling

### Database Errors
- Connection timeout handling
- Query error recovery
- Graceful degradation

### API Errors
- OpenAI API error handling
- Rate limiting management
- Fallback mechanisms

### Data Validation
- Input validation
- Response format verification
- Missing data handling

## 📝 Troubleshooting

### Common Issues

1. **Database Connection Failed**
   - Verify database credentials in `.env`
   - Check network connectivity
   - Ensure database server is running

2. **No Questions Found**
   - Verify Survey ID exists
   - Check for open-ended questions (type 'T')
   - Ensure questions have parent_qid = 0

3. **OpenAI API Errors**
   - Verify API key is valid
   - Check API quota and billing
   - Ensure internet connectivity

4. **Empty Responses**
   - Check JSON column structure
   - Verify question ID mapping
   - Review data filtering logic

### Debug Mode
Enable debug logging by setting:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a virtual environment
3. Install development dependencies
4. Make your changes
5. Test thoroughly
6. Submit a pull request

### Code Style
- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add docstrings for functions and classes
- Include error handling

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
1. Check the troubleshooting section
2. Review the code documentation
3. Create an issue with detailed information
4. Include error logs and configuration details

## 🔮 Future Enhancements

### Planned Features
- Support for additional LLM providers
- Advanced sentiment analysis
- Automated report generation
- Integration with PowerPoint/Excel
- Real-time collaboration features
- Custom category templates
- Multi-language support

### Performance Improvements
- Caching mechanisms
- Parallel processing
- Database optimization
- API call optimization

---

**Built for pharmaceutical market research professionals who need powerful, AI-driven insights from qualitative survey data.**
