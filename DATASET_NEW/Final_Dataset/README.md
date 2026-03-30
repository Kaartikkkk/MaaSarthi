# MaaSarthi Master Dataset v1.0

## 📊 Dataset Overview

**Purpose**: Comprehensive job dataset optimized for Indian women's employment and career development

**Total Records**: 328,077
- Real Data: 328,077 records (100.0%)
- Synthetic Data: 0 records (0.0%)

**Features**: 48 columns covering job details, salaries, requirements, and MaaSarthi-specific metrics

## 🎯 Key Features

### Job Information
- **job_title**: Position titles
- **company**: Employer names  
- **location**: Job locations (India-focused)
- **job_category**: Categorized job types

### Salary Details
- **salary_min/max/avg**: Comprehensive salary information in INR
- **salary_bracket**: Categorized salary ranges (LPA format)

### Experience Requirements
- **required_experience_min/max**: Experience requirements in years
- **experience_category**: Categorized experience levels

### Work Flexibility
- **remote_flag**: Remote work availability
- **work_arrangement**: Work mode options
- **flexible_hours**: Flexible timing availability

### MaaSarthi-Specific Features
- **female_friendly**: Gender-inclusive workplace indicator
- **mother_suitability_score**: Working mothers compatibility (0-10)
- **childcare_support**: Childcare assistance availability

## 📈 Dataset Statistics

### Job Distribution
- India-based jobs: 681 (0.2%)
- Remote jobs: 32,507 (9.9%)
- Female-friendly: 67,813 (20.7%)

### Data Quality
- Completeness: 79.5%
- Duplicates: 0
- Memory Usage: 491.7 MB

## 🚀 Usage Guidelines

### Primary Use Cases
1. **Job Recommendation Systems**: ML models for personalized job matching
2. **Salary Prediction**: Predict salary ranges based on skills and experience
3. **Market Analysis**: Analyze job market trends for women
4. **Skills Gap Analysis**: Identify skill requirements and gaps

### Preprocessing Recommendations
1. Handle categorical variables with appropriate encoding
2. Scale numerical features for ML models
3. Consider stratified sampling for class imbalance
4. Feature engineering for domain-specific insights

## 📚 Column Reference

See `dataset_documentation.json` for detailed column descriptions and metadata.

## 📋 Data Processing Pipeline

This dataset was created through a comprehensive 13-phase processing pipeline:

1. **Phase 3**: Dataset Inspection
2. **Phase 4**: Column Standardization  
3. **Phase 5**: Data Cleaning
4. **Phase 6**: Salary Normalization
5. **Phase 7**: Experience Extraction
6. **Phase 8**: Feature Creation
7. **Phase 9**: Dataset Merging
8. **Phase 10**: Duplicate Removal
9. **Phase 11**: Missing Data Handling
10. **Phase 12**: Dataset Validation
11. **Phase 13**: Master Dataset Creation

## ⚠️ Important Notes

- This dataset is optimized for Indian job market analysis
- Synthetic data follows realistic patterns based on real job market data
- All salary figures are in Indian Rupees (INR) per annum
- Focus on flexible and women-friendly employment opportunities

## 📊 Version History

- **v1.0.0** (2026-03-04): Initial release with 328,077 records

---

Created by MaaSarthi Data Team | March 2026
