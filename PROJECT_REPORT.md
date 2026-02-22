# MaaSarthi - Project Report

## 📌 Executive Summary

**MaaSarthi** (meaning "Mother's Companion" in Hindi) is an AI-powered career platform specifically designed to empower women, particularly mothers and homemakers, by connecting them with flexible work-from-home opportunities and skill development resources. The platform uses machine learning to provide personalized job recommendations based on individual skills, education, availability, and preferences.

---

## 🎯 Problem Statement

Millions of women in India, especially mothers, face challenges in:
- Finding flexible work that accommodates family responsibilities
- Identifying jobs matching their existing skills
- Accessing skill training to increase earning potential
- Navigating complex job portals

**MaaSarthi solves these problems** by providing a simple, personalized, and AI-driven platform tailored for women seeking work-from-home opportunities.

---

## 🌟 Key Features

### 1. AI-Powered Job Recommendations
- Machine learning model analyzes user profile (age, education, skills, domain expertise)
- Provides personalized job matches with confidence scores
- Estimates salary ranges using quantile-based calculations
- Direct links to apply on LinkedIn, Naukri, Indeed, and Internshala

### 2. Skill Development Platform
- Curated skill training recommendations
- Progress tracking for ongoing skill development
- Resources and courses for popular skills (data entry, content writing, graphic design, etc.)

### 3. User Dashboard
- Personal overview with statistics
- Task management system
- Skill progress tracking
- Reminder notifications

### 4. AI Career Assistant
- Intelligent chatbot for career guidance
- Answers questions about jobs, skills, and opportunities
- Available 24/7 on every page

### 5. Secure Authentication
- Secure signup/login with password hashing
- Profile management with dropdown menu
- Session-based authentication with automatic expiry

---

## 🛠️ Technology Stack

### Backend
| Technology | Purpose |
|------------|---------|
| **Python 3.14** | Primary programming language |
| **Flask** | Web framework for routing and API |
| **SQLAlchemy** | ORM for database operations |
| **SQLite** | Lightweight relational database |
| **Jinja2** | Template engine for HTML rendering |

### Machine Learning
| Technology | Purpose |
|------------|---------|
| **scikit-learn** | ML library for model training |
| **RandomForestClassifier** | Job recommendation algorithm |
| **TF-IDF Vectorizer** | Text feature extraction |
| **pandas** | Data manipulation and analysis |
| **NumPy** | Numerical computations |

### Frontend
| Technology | Purpose |
|------------|---------|
| **HTML5** | Page structure |
| **CSS3** | Styling with modular architecture |
| **JavaScript** | Interactive functionality |
| **SVG Icons** | Professional iconography |

### Security
| Feature | Implementation |
|---------|----------------|
| Password Hashing | SHA-256 with unique salt per user |
| Rate Limiting | 5 login attempts per 5 minutes |
| Session Security | HTTPOnly cookies, SameSite=Lax |
| Input Sanitization | XSS and injection prevention |
| CSRF Protection | SameSite cookie policy |

---

## 📊 Dataset Information

### Source
The training dataset (`dataset.csv`) contains job and skill data curated for women-focused work-from-home opportunities.

### Dataset Structure
| Column | Description | Data Type |
|--------|-------------|-----------|
| `Age` | Age group of candidate | Integer |
| `Education` | Highest qualification (10th, 12th, Graduate, Post Graduate) | Categorical |
| `Domain` | Area of expertise | Categorical |
| `Skill` | Primary skill | Categorical |
| `Work Mode` | Preferred work type (Remote, Hybrid, On-site) | Categorical |
| `Location` | Geographic preference | Categorical |
| `Job Title` | Recommended job role | Categorical |
| `Company` | Potential employer | String |
| `Salary` | Expected salary range | String |
| `Description` | Job description | Text |

### Skill Categories
- **Technical**: Data Entry, Content Writing, Graphic Design, Web Development
- **Creative**: Photography, Video Editing, Social Media Management
- **Service**: Online Tutoring, Virtual Assistant, Customer Support
- **Traditional**: Tailoring, Cooking, Handicrafts, Beauty Services

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      USER INTERFACE                          │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────────┐ │
│  │  Home   │  │  Jobs   │  │ Skills  │  │    Dashboard    │ │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────────┬────────┘ │
└───────┼────────────┼────────────┼────────────────┼──────────┘
        │            │            │                │
        ▼            ▼            ▼                ▼
┌─────────────────────────────────────────────────────────────┐
│                    FLASK APPLICATION                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │   Routing    │  │   Security   │  │   ML Prediction  │   │
│  │   (app.py)   │  │  (Auth/Rate) │  │  (RandomForest)  │   │
│  └──────────────┘  └──────────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────────┘
        │                    │                    │
        ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│    SQLite DB    │  │   Session Mgmt  │  │   Dataset CSV   │
│  (maasarthi.db) │  │   (Flask-Sess)  │  │  (dataset.csv)  │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

---

## 📁 Project Structure

```
MaaSarthi/
│
├── app.py                    # Main Flask application (2000+ lines)
├── train_model.py            # ML model training script
├── dataset.csv               # Training dataset
├── maasarthi.db             # SQLite database
├── start_server.sh          # Server startup script
├── performance_test.py      # Performance testing utilities
│
├── static/
│   ├── css/
│   │   ├── base.css         # Global styles, CSS variables
│   │   ├── navbar.css       # Navigation + profile dropdown
│   │   ├── home.css         # Home page styling
│   │   ├── login.css        # Login page
│   │   ├── signup.css       # Signup page
│   │   ├── form.css         # Job/skill forms
│   │   ├── result.css       # Results with job cards
│   │   ├── dashboard.css    # User dashboard
│   │   ├── chatbot.css      # AI chatbot widget
│   │   ├── footer.css       # Footer component
│   │   ├── contact.css      # Contact page
│   │   ├── animations.css   # Animation keyframes
│   │   └── assistant.css    # AI assistant
│   │
│   ├── js/
│   │   └── animations.js    # Scroll & page animations
│   │
│   └── images/
│       └── logo.png         # MaaSarthi logo
│
└── templates/
    ├── home.html            # Landing page
    ├── login.html           # User login
    ├── signup.html          # User registration
    ├── form.html            # Job search form
    ├── result.html          # Job recommendations
    ├── skill_form.html      # Skill selection
    ├── skill_result.html    # Skill recommendations
    ├── dashboard.html       # User dashboard
    ├── contact.html         # Contact form
    ├── chatbot.html         # Chatbot component
    └── assistant.html       # AI assistant page
```

---

## 🔐 Database Schema

### Users Table
```sql
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(120) UNIQUE NOT NULL,
    phone VARCHAR(15),
    password_hash VARCHAR(256) NOT NULL,
    salt VARCHAR(64) NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_login DATETIME,
    is_active BOOLEAN DEFAULT TRUE,
    is_verified BOOLEAN DEFAULT FALSE
);
```

### Tasks Table
```sql
CREATE TABLE tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    title VARCHAR(200) NOT NULL,
    description TEXT,
    status VARCHAR(20) DEFAULT 'pending',
    due_date DATETIME,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);
```

### User Skills Table
```sql
CREATE TABLE user_skills (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    skill_name VARCHAR(100) NOT NULL,
    proficiency INTEGER DEFAULT 0,
    progress INTEGER DEFAULT 0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);
```

### Contact Messages Table
```sql
CREATE TABLE contact_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(120) NOT NULL,
    subject VARCHAR(200),
    message TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    is_read BOOLEAN DEFAULT FALSE
);
```

### User Profiles Table
```sql
CREATE TABLE user_profiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL UNIQUE,
    age INTEGER,
    education VARCHAR(50),
    city_type VARCHAR(50),
    language VARCHAR(50),
    device VARCHAR(50),
    location VARCHAR(100),
    work_preference VARCHAR(50),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME,
    FOREIGN KEY (user_id) REFERENCES users(id)
);
```

### Job Search History Table
```sql
CREATE TABLE job_search_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    age INTEGER,
    kids INTEGER,
    hours INTEGER,
    domain VARCHAR(100),
    skill VARCHAR(100),
    education VARCHAR(50),
    city_type VARCHAR(50),
    language VARCHAR(50),
    device VARCHAR(50),
    work_mode VARCHAR(50),
    location VARCHAR(100),
    predicted_work VARCHAR(200),
    predicted_salary_low INTEGER,
    predicted_salary_high INTEGER,
    confidence_score FLOAT,
    search_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);
```

### Skill Search History Table
```sql
CREATE TABLE skill_search_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    skill_name VARCHAR(100),
    skill_level VARCHAR(50),
    hours_available INTEGER,
    estimated_income_low INTEGER,
    estimated_income_high INTEGER,
    search_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);
```

---

## 🤖 Machine Learning Pipeline

### 1. Data Preprocessing
```python
# Load and clean dataset
df = pd.read_csv('dataset.csv')

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(text_features)
```

### 2. Model Training
```python
# RandomForest Classifier
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
model.fit(X_train, y_train)
```

### 3. Prediction Process
```python
# User input processing
user_features = f"{age} {education} {skill} {domain} {work_mode}"
user_vector = vectorizer.transform([user_features])

# Get prediction with confidence
prediction = model.predict(user_vector)
confidence = model.predict_proba(user_vector).max() * 100
```

### 4. Salary Estimation
```python
# Quantile-based salary range calculation
salary_data = df[df['Job Title'] == predicted_job]['Salary']
low = salary_data.quantile(0.25)
high = salary_data.quantile(0.75)
```

---

## 🌐 API Endpoints

### Public Routes
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Home page |
| `/login` | GET, POST | User authentication |
| `/signup` | GET, POST | User registration |
| `/logout` | GET | End user session |
| `/contact` | GET, POST | Contact form |
| `/chat` | POST | AI chatbot responses |

### Protected Routes (Login Required)
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/jobs` | GET | Job search form |
| `/predict` | POST | Get job recommendations |
| `/skills` | GET | Skill selection form |
| `/skills-result` | POST | Get skill recommendations |
| `/dashboard` | GET | User dashboard |
| `/find-jobs-nearby` | GET | External job search |

### Admin Routes
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/admin/retrain-model` | POST | Retrain ML model |
| `/admin/model-stats` | GET | Model statistics |

### User History & Profile APIs
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/job-history` | GET | Get user's job search history |
| `/api/skill-history` | GET | Get user's skill search history |
| `/api/user-profile` | GET | Get user profile information |
| `/api/update-profile` | POST | Update user profile |
| `/api/dashboard-stats` | GET | Get dashboard statistics |

### Task & Reminder APIs
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/add-task` | POST | Create new task |
| `/api/toggle-task/<id>` | POST | Toggle task completion |
| `/api/delete-task/<id>` | DELETE | Delete a task |
| `/api/add-reminder` | POST | Create new reminder |
| `/api/toggle-reminder/<id>` | POST | Toggle reminder status |
| `/api/delete-reminder/<id>` | DELETE | Delete a reminder |

### Skill Progress APIs
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/add-skill` | POST | Add skill to profile |
| `/api/update-skill-progress/<id>` | PUT | Update skill progress |
| `/api/delete-skill/<id>` | DELETE | Remove skill from profile |

---

## 🎨 UI/UX Design

### Color Palette
| Color | Hex Code | Usage |
|-------|----------|-------|
| Primary Pink | `#ec4899` | Buttons, accents, highlights |
| Dark Pink | `#db2777` | Hover states, gradients |
| Light Pink | `#fdf2f8` | Backgrounds |
| Pale Pink | `#fce7f3` | Card backgrounds |
| Dark Gray | `#1f2937` | Text |
| Medium Gray | `#6b7280` | Secondary text |

### Typography
- **Primary Font**: Inter (Google Fonts)
- **Weights**: 300, 400, 500, 600, 700

### Components
- **Navbar**: Fixed position with profile dropdown
- **Cards**: Rounded corners (16-24px), subtle shadows
- **Buttons**: Gradient backgrounds, hover animations
- **Forms**: Modern inputs with icon labels
- **Animations**: Fade-in, slide-up, scroll-triggered

---

## 🚀 Deployment Instructions

### Local Development
```bash
# Clone repository
git clone https://github.com/Kaartikkkk/MaaSarthi.git
cd MaaSarthi

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate     # Windows

# Install dependencies
pip install flask flask-sqlalchemy pandas scikit-learn numpy

# Run server
python app.py
```

### Access Application
- **URL**: http://127.0.0.1:5001
- **Default Port**: 5001

---

## 📈 Future Enhancements

1. **Mobile App** - React Native/Flutter mobile application
2. **Multi-language Support** - Hindi, Tamil, Telugu, etc.
3. **Video Courses** - Integrated skill training videos
4. **Employer Portal** - Companies can post jobs directly
5. **Resume Builder** - AI-powered resume creation
6. **Payment Integration** - Premium features and course purchases
7. **Community Forum** - Peer support and networking
8. **Push Notifications** - Job alerts and reminders

---

## 👥 Target Audience

- **Primary**: Mothers seeking work-from-home opportunities
- **Secondary**: Homemakers wanting to develop skills
- **Tertiary**: Women returning to workforce after career break

---

## 📊 Impact Metrics (Projected)

| Metric | Target |
|--------|--------|
| Users Registered | 10,000+ |
| Jobs Recommended | 50,000+ |
| Skills Trained | 15,000+ |
| Average Monthly Income | ₹25,000 |
| Partner Companies | 500+ |

---

## 📝 License

This project is developed for educational and social impact purposes.

---

## 📞 Contact

- **Email**: support@maasarthi.com
- **GitHub**: https://github.com/Kaartikkkk/MaaSarthi

---

*Report Generated: February 22, 2026*
*Version: 1.0*
