# MaaSarthi - Women Empowerment Platform

## 🎯 Project Overview

**MaaSarthi** is a comprehensive AI-powered career guidance and skill training platform designed specifically for mothers and women seeking work-from-home opportunities, skill development, and financial independence. The platform provides personalized job recommendations, income predictions, and curated learning roadmaps based on individual profiles.

---

## ✨ Key Features

### 1. **Intelligent Job Recommendation System**
- Personalized work suggestions based on:
  - Age, education level, and family responsibilities
  - Available hours per day
  - Skills and domain expertise
  - Location and work mode preferences
  - Device availability (Mobile/Laptop)
  - Language proficiency (Hindi/English/Both)
- Expected monthly income range predictions
- ML-powered recommendations using Random Forest models

### 2. **Skill Training Platform**
- 25+ skill categories including:
  - Creative skills: Cooking, Baking, Beauty, Mehndi, Handicraft
  - Digital skills: Data Entry, Excel, Canva, Social Media, Video Editing
  - Technical skills: Graphic Design, Content Writing
  - Service skills: Teaching, Babysitting, Home Tutor, Caregiver
  - Trade skills: Tailoring, Electrician, Plumbing, Mobile Repairing
- Customized learning roadmaps
- Curated resources from YouTube, Google, and Instagram
- Progress tracking and portfolio building guidance

### 3. **AI-Powered Chatbot Assistant**
- Real-time conversational support using OpenAI GPT-4
- Bilingual support (Hindi & English)
- Context-aware responses for:
  - Job queries and recommendations
  - Skill learning paths
  - Income estimates
  - Training roadmaps
- Dataset-integrated smart suggestions

### 4. **Bilingual Interface**
- Complete Hindi and English language support
- Easy language toggle
- Culturally relevant content and terminology

### 5. **Income Prediction Engine**
- Data-driven salary estimates
- Factors considered:
  - Skill level (Beginner/Intermediate/Advanced)
  - Hours commitment
  - Domain and education
  - Geographic location type (Urban/Semi-Urban/Rural)

---

## 🛠️ Technical Stack

### **Backend**
- **Framework**: Flask (Python)
- **Machine Learning**: 
  - scikit-learn (Random Forest Classifier & Regressor)
  - Preprocessing: OneHotEncoder, ColumnTransformer
  - Model persistence: joblib
- **AI Integration**: OpenAI GPT-4 API
- **Data Processing**: pandas, numpy

### **Frontend**
- **HTML5** with Jinja2 templating
- **CSS3** with custom responsive design
- **Vanilla JavaScript** for interactivity
- Modern UI/UX with gradient designs and smooth animations

### **Data Storage**
- CSV-based dataset (`dataset.csv`)
- Session management for user preferences
- Environment variables for API keys (.env)

### **Deployment Ready**
- Environment variable configuration
- Safe model loading with fallback mechanisms
- Error handling and logging

---

## 📁 Project Structure

```
MaaSarthi/
│
├── app.py                      # Main Flask application
├── train_model.py              # ML model training script
├── dataset.csv                 # Training dataset
├── work_model.pkl              # Trained job recommendation model
├── income_model.pkl            # Trained income prediction model
│
├── templates/
│   ├── home.html              # Landing page
│   ├── form.html              # Job recommendation form
│   ├── skill_form.html        # Skill training form
│   ├── result.html            # Job recommendation results
│   ├── skill_result.html      # Skill training results
│   ├── chatbot.html           # Chatbot widget component
│   └── job_form.html          # Alternative job form
│
├── static/
│   ├── style.css              # Main stylesheet
│   ├── favicon.png            # App icon
│   ├── favicon.ico            # App icon (ICO format)
│   └── images/
│       └── logo.png           # Brand logo
│
├── .env                       # Environment variables (API keys)
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## 🤖 Machine Learning Models

### **Work Recommendation Model (Classification)**
- **Algorithm**: Random Forest Classifier
- **Parameters**:
  - 600 estimators
  - Balanced class weights
  - Multi-core processing (n_jobs=-1)
- **Features**: Age, Kids, Hours, Domain, Skill, Education, City Type, Language, Device, Work Mode
- **Target**: Work Type recommendation

### **Income Prediction Model (Regression)**
- **Algorithm**: Random Forest Regressor
- **Parameters**:
  - 600 estimators
  - Optimized for numerical prediction
- **Features**: Same as classification model
- **Target**: Monthly income prediction

### **Preprocessing Pipeline**
- Categorical encoding with unknown value handling
- Numeric feature pass-through
- Safe handling of new/unseen categories

---

## 🎨 User Interface Highlights

### **Home Page**
- Hero section with gradient design
- Feature cards showcasing platform benefits
- Language switcher (English/Hindi)
- Quick access to Jobs and Skills sections
- Integrated chatbot widget

### **Job Recommendation Flow**
1. User fills detailed profile form
2. ML model processes inputs
3. Results display:
   - Recommended work type
   - Income range (₹ low - ₹ high)
   - Next skills to learn
   - Helpful resource links

### **Skill Training Flow**
1. Skill selection with 25+ options
2. Dynamic skill information cards
3. Level, hours, goal, and learning mode selection
4. Results include:
   - Step-by-step roadmap
   - Curated learning resources
   - Expected income potential
   - Job opportunity links

### **Chatbot Interface**
- Fixed position floating widget
- Minimize/maximize functionality
- Clean, modern chat bubbles
- Real-time typing indicators
- Smooth animations

---

## 🚀 Setup Instructions

### **Prerequisites**
```bash
Python 3.8+
pip (Python package manager)
OpenAI API key
```

### **Installation**

1. **Clone the repository**
```bash
git clone 
cd MaaSarthi
```

2. **Install dependencies**
```bash
pip install flask pandas scikit-learn joblib python-dotenv openai
```

Or use requirements.txt:
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_openai_api_key_here
```

4. **Prepare your dataset**
Ensure `dataset.csv` exists with the required columns (see Dataset Structure below)

5. **Train the models**
```bash
python train_model.py
```
This generates `work_model.pkl` and `income_model.pkl`

6. **Run the application**
```bash
python app.py
```

7. **Access the platform**
Open your browser and navigate to:
```
http://localhost:5000
```

---

## 📊 Dataset Structure

The `dataset.csv` should contain the following columns:

| Column | Type | Description | Example Values |
|--------|------|-------------|----------------|
| age | Integer | User's age | 25, 30, 35 |
| kids | Integer | Number of children | 0, 1, 2, 3 |
| hours | Integer | Available hours per day | 1, 2, 3, 4, 5 |
| domain | String | Work domain/field | Cooking, IT, Teaching |
| skill | String | Primary skill | Excel, Baking, Data Entry |
| education | String | Education level | 10th, 12th, UG, PG |
| city_type | String | Location type | Urban, Semi-Urban, Rural |
| language | String | Language proficiency | Hindi, English, Both |
| device | String | Device availability | Mobile, Laptop, Both |
| work_mode | String | Preferred work mode | Work From Home, Hybrid, Offline Local |
| work_type | String | **Target** - Recommended work | Freelancing, Job, Business |
| income | Integer | **Target** - Monthly income | 5000, 10000, 15000 |

### Sample Dataset Entry:
```csv
age,kids,hours,domain,skill,education,city_type,language,device,work_mode,work_type,income
28,2,3,Cooking,Baking,12th,Urban,Hindi,Mobile,Work From Home,Freelancing,8000
```

---

## 🌐 API Endpoints

### **Core Routes**
- `GET /` - Home page
- `GET /jobs` - Job recommendation form
- `POST /predict` - Generate job recommendations
- `GET /skills` - Skill training form
- `POST /skills-result` - Generate skill training plan
- `POST /chat` - Chatbot API endpoint
- `GET /set-language/<lang>` - Switch language (en/hi)
- `GET /find-jobs-nearby` - Redirect to job search with location

### **Static Routes**
- `GET /favicon.ico` - Favicon delivery
- `GET /static/<path>` - Static file serving

---

## 🔒 Security Features

- Environment-based API key management
- Safe model loading with error handling
- Input validation and sanitization
- Session-based user data storage
- No sensitive data in frontend
- CSRF protection (Flask session secret)

---

## 🎯 Target Audience

- Mothers seeking work-from-home opportunities
- Women looking to develop new skills
- Individuals with family responsibilities
- First-time job seekers
- Career changers
- Rural and semi-urban women
- Non-technical users
- Hindi and English speakers

---

## 💡 Unique Value Propositions

1. **Culturally Sensitive**: Designed specifically for Indian mothers with bilingual support
2. **Realistic Expectations**: Data-driven income predictions, not promises
3. **Skill-First Approach**: Focus on learning and capability building
4. **Accessibility**: Mobile-friendly, works with basic devices
5. **Comprehensive**: Job finding + skill training in one platform
6. **AI-Powered**: Smart recommendations and conversational support
7. **Free Resources**: Curated links to free learning materials
8. **No Registration Required**: Immediate access to core features

---

## 🔧 Configuration

### Environment Variables (.env)
```bash
OPENAI_API_KEY=sk-your-api-key-here
FLASK_SECRET_KEY=maasarthi_secret_key_123  # Optional: customize
FLASK_ENV=development  # or production
```

### Model Configuration (train_model.py)
You can adjust these parameters:
- `n_estimators`: Number of trees (default: 600)
- `random_state`: For reproducibility (default: 42)
- `class_weight`: For imbalanced data (default: "balanced_subsample")

---

## 🐛 Troubleshooting

### Common Issues

1. **Models not loading**
   - Ensure `train_model.py` has been run successfully
   - Check that `.pkl` files exist in the project root
   - Verify dataset.csv is properly formatted

2. **Chatbot not responding**
   - Verify OpenAI API key in `.env` file
   - Check API credit balance
   - Review browser console for JavaScript errors

3. **Language not switching**
   - Clear browser cache
   - Check session management in browser
   - Ensure cookies are enabled

4. **Income predictions seem off**
   - Retrain models with more diverse dataset
   - Check for outliers in training data
   - Verify all features are being passed correctly

---

## 📈 Future Enhancements

- [ ] User authentication and profile saving
- [ ] Progress tracking dashboard
- [ ] Community forum for peer support
- [ ] Video tutorials integrated
- [ ] Certification partnerships
- [ ] Direct employer connections
- [ ] Mobile app version (React Native/Flutter)
- [ ] Regional language support (Tamil, Telugu, Bengali, etc.)
- [ ] Advanced analytics dashboard
- [ ] Success stories section
- [ ] Payment integration for premium features
- [ ] Mentor matching system
- [ ] Job application tracking
- [ ] Skill assessment tests

---

## 🤝 Contributing

This project aims to empower women through technology. Contributions are welcome!

### How to Contribute:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Areas for Contribution:
- Dataset expansion with real-world data
- UI/UX improvements
- Additional language support
- Feature enhancements
- Bug fixes
- Documentation improvements
- Test coverage
- Performance optimization

---

## 📞 Support

### For Technical Issues:
- Check console logs for debugging
- Ensure API keys are correctly configured
- Verify dataset format matches requirements
- Test that models are properly trained
- Review Flask logs for backend errors

### For Feature Requests:
- Open an issue on GitHub
- Provide detailed use case
- Include mockups if applicable

---

## 📄 License

This project is built for social impact and women empowerment. 

**License Type**: MIT License (or specify your chosen license)

```
Copyright (c) 2024 MaaSarthi Project

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

## 🙏 Acknowledgments

- Built with the vision of creating economic opportunities for mothers and women across India
- Inspired by the need to balance family responsibilities with financial independence
- Powered by open-source technologies and community support
- Special thanks to all contributors and testers

---

## 📊 Project Statistics

- **Skills Supported**: 25+
- **Languages**: 2 (Hindi, English)
- **Work Modes**: 3 (Work From Home, Hybrid, Offline)
- **Education Levels**: 7 (No Formal to Post Graduate)
- **ML Models**: 2 (Classification + Regression)
- **Total Code Files**: 15+

---

## 🌟 Impact Goals

- Empower 10,000+ women in the first year
- Create awareness about work-from-home opportunities
- Bridge the digital skills gap
- Provide accessible career guidance
- Build a supportive community
- Enable financial independence for mothers

---

## 📚 Additional Resources

### For Users:
- [How to use MaaSarthi - Video Tutorial](#)
- [Skill Training Best Practices](#)
- [Work From Home Success Stories](#)

### For Developers:
- [Flask Documentation](https://flask.palletsprojects.com/)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [OpenAI API Documentation](https://platform.openai.com/docs)

---

## 🔗 Quick Links

- **Demo**: [Live Demo Link](#)
- **Documentation**: [Full Documentation](#)
- **GitHub**: [Repository Link](#)
- **Feedback**: [Feedback Form](#)
- **Contact**: contact@maasarthi.com

---

**MaaSarthi** - *Empowering Mothers Through Skills and Opportunities* 🌟

---

## Version History

### v1.0.0 (Current)
- Initial release
- Job recommendation system
- Skill training platform
- AI chatbot integration
- Bilingual support (Hindi/English)
- Income prediction
- ML-powered recommendations

### Planned for v1.1.0
- User authentication
- Progress tracking
- Enhanced chatbot capabilities
- Mobile responsiveness improvements
- Additional regional languages

---

**Last Updated**: January 2026
**Maintained By**: MaaSarthi Development Team