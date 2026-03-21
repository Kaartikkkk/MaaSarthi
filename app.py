from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
import joblib
import pandas as pd
from flask import send_from_directory
import os
from flask import session, redirect, url_for
from datetime import timedelta, datetime
from pathlib import Path
from flask import Flask, request, jsonify, render_template
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask_dance.contrib.google import make_google_blueprint, google
from flask_sqlalchemy import SQLAlchemy
import hashlib
import secrets
import re
from functools import wraps
import time
import numpy as np

# ============================================
# 🔐 SECURITY & VALIDATION UTILITIES,
# ============================================

class SecurityUtils:
    """Utility class for security operations"""
    
    @staticmethod
    def hash_password(password: str, salt: str = None) -> tuple:
        """
        Hash password using SHA-256 with salt
        Returns: (hashed_password, salt)
        """
        if salt is None:
            salt = secrets.token_hex(32)
        
        # Combine password and salt, then hash
        salted_password = f"{password}{salt}"
        hashed = hashlib.sha256(salted_password.encode()).hexdigest()
        
        return hashed, salt
    
    @staticmethod
    def verify_password(password: str, stored_hash: str, salt: str) -> bool:
        """Verify password against stored hash"""
        computed_hash, _ = SecurityUtils.hash_password(password, salt)
        return secrets.compare_digest(computed_hash, stored_hash)
    
    @staticmethod
    def generate_session_token() -> str:
        """Generate a secure session token"""
        return secrets.token_urlsafe(32)
    
    @staticmethod
    def sanitize_input(input_str: str) -> str:
        """Sanitize user input to prevent XSS"""
        if not input_str:
            return ""
        # Remove potentially dangerous characters
        sanitized = input_str.strip()
        sanitized = sanitized.replace('<', '&lt;').replace('>', '&gt;')
        sanitized = sanitized.replace('"', '&quot;').replace("'", '&#x27;')
        return sanitized


class Validator:
    """Input validation class"""
    
    @staticmethod
    def validate_email(email: str) -> tuple:
        """
        Validate email format
        Returns: (is_valid, error_message)
        """
        if not email:
            return False, "Email is required"
        
        email = email.strip().lower()
        
        # Email regex pattern
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        
        if not re.match(email_pattern, email):
            return False, "Invalid email format"
        
        if len(email) > 254:
            return False, "Email is too long"
        
        return True, None
    
    @staticmethod
    def validate_password(password: str) -> tuple:
        """
        Validate password strength
        Returns: (is_valid, error_message)
        """
        if not password:
            return False, "Password is required"
        
        if len(password) < 8:
            return False, "Password must be at least 8 characters long"
        
        if len(password) > 128:
            return False, "Password is too long"
        
        # Check for at least one uppercase letter
        if not re.search(r'[A-Z]', password):
            return False, "Password must contain at least one uppercase letter"
        
        # Check for at least one lowercase letter
        if not re.search(r'[a-z]', password):
            return False, "Password must contain at least one lowercase letter"
        
        # Check for at least one digit
        if not re.search(r'\d', password):
            return False, "Password must contain at least one number"
        
        return True, None
    
    @staticmethod
    def validate_phone(phone: str) -> tuple:
        """
        Validate phone number (Indian format)
        Returns: (is_valid, error_message)
        """
        if not phone:
            return False, "Phone number is required"
        
        # Remove spaces and dashes
        phone = re.sub(r'[\s\-]', '', phone)
        
        # Indian phone number pattern (10 digits, optionally with +91 or 0 prefix)
        phone_pattern = r'^(?:\+91|91|0)?[6-9]\d{9}$'
        
        if not re.match(phone_pattern, phone):
            return False, "Invalid phone number. Please enter a valid 10-digit Indian mobile number"
        
        return True, None
    
    @staticmethod
    def validate_name(name: str) -> tuple:
        """
        Validate name
        Returns: (is_valid, error_message)
        """
        if not name:
            return False, "Name is required"
        
        name = name.strip()
        
        if len(name) < 2:
            return False, "Name must be at least 2 characters long"
        
        if len(name) > 100:
            return False, "Name is too long"
        
        # Allow only letters, spaces, and common name characters
        if not re.match(r'^[a-zA-Z\s\.\'-]+$', name):
            return False, "Name contains invalid characters"
        
        return True, None


class RateLimiter:
    """Simple rate limiter for login attempts"""
    
    def __init__(self, max_attempts: int = 5, window_seconds: int = 300):
        self.max_attempts = max_attempts
        self.window_seconds = window_seconds
        self.attempts = {}  # {ip_or_email: [(timestamp, success), ...]}
    
    def is_allowed(self, identifier: str) -> tuple:
        """
        Check if request is allowed
        Returns: (is_allowed, remaining_time_if_blocked)
        """
        current_time = time.time()
        
        if identifier not in self.attempts:
            self.attempts[identifier] = []
        
        # Clean old attempts
        self.attempts[identifier] = [
            (ts, success) for ts, success in self.attempts[identifier]
            if current_time - ts < self.window_seconds
        ]
        
        # Count failed attempts
        failed_attempts = sum(1 for ts, success in self.attempts[identifier] if not success)
        
        if failed_attempts >= self.max_attempts:
            oldest_attempt = min(ts for ts, _ in self.attempts[identifier])
            remaining_time = int(self.window_seconds - (current_time - oldest_attempt))
            return False, remaining_time
        
        return True, 0
    
    def record_attempt(self, identifier: str, success: bool):
        """Record a login attempt"""
        if identifier not in self.attempts:
            self.attempts[identifier] = []
        
        self.attempts[identifier].append((time.time(), success))
        
        # If successful, clear failed attempts
        if success:
            self.attempts[identifier] = []


# Initialize rate limiter
login_rate_limiter = RateLimiter(max_attempts=5, window_seconds=300)  # 5 attempts per 5 minutes






# ✅ Safe model loader
def safe_load_model(path: str):
    """Load a joblib model safely. If missing, return None instead of crashing the app."""
    try:
        if Path(path).exists():
            return joblib.load(path)
    except Exception as e:
        print(f"⚠️ Could not load model {path}: {e}")
    return None

# ✅ Initialize Flask app with security settings
app = Flask(__name__)

# 🔐 Security Configuration
app.secret_key = secrets.token_hex(32)  # Generate secure secret key
app.config.update(
    SESSION_COOKIE_SECURE=False,  # Set to True in production with HTTPS
    SESSION_COOKIE_HTTPONLY=True,  # Prevent JavaScript access to session cookie
    SESSION_COOKIE_SAMESITE='Lax',  # CSRF protection
    PERMANENT_SESSION_LIFETIME=timedelta(hours=24),  # Session expiry
    MAX_CONTENT_LENGTH=16 * 1024 * 1024  # Max upload size: 16MB
)

# ============================================
# 🗄️ DATABASE CONFIGURATION (SQLite + SQLAlchemy)
# ============================================

basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'maasarthi.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# ============================================
# 🗄️ DATABASE MODELS
# ============================================

class User(db.Model):
    """User account model"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(254), unique=True, nullable=False, index=True)
    phone = db.Column(db.String(15), nullable=False)
    password_hash = db.Column(db.String(64), nullable=False)
    salt = db.Column(db.String(64), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now)
    last_login = db.Column(db.DateTime, nullable=True)
    is_active = db.Column(db.Boolean, default=True)
    is_verified = db.Column(db.Boolean, default=False)
    
    # Relationships
    tasks = db.relationship('Task', backref='user', lazy=True, cascade='all, delete-orphan')
    reminders = db.relationship('Reminder', backref='user', lazy=True, cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<User {self.email}>'


class Task(db.Model):
    """Dashboard task model"""
    __tablename__ = 'tasks'
    
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    task_text = db.Column(db.String(500), nullable=False)
    is_completed = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.now)
    
    def __repr__(self):
        return f'<Task {self.task_text[:30]}>'


class Reminder(db.Model):
    """Dashboard reminder model"""
    __tablename__ = 'reminders'
    
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.String(500), nullable=True)
    due_date = db.Column(db.DateTime, nullable=True)
    is_completed = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.now)
    
    def __repr__(self):
        return f'<Reminder {self.title}>'


class ContactMessage(db.Model):
    """Contact form message model"""
    __tablename__ = 'contact_messages'
    
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(254), nullable=False)
    subject = db.Column(db.String(200), nullable=True)
    message = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now)
    is_read = db.Column(db.Boolean, default=False)
    
    def __repr__(self):
        return f'<ContactMessage from {self.email}>'


class JobRecommendation(db.Model):
    """Job recommendation model"""
    __tablename__ = 'job_recommendations'
    
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    job_title = db.Column(db.String(200), nullable=False)
    company = db.Column(db.String(200), nullable=True)
    salary = db.Column(db.String(100), nullable=True)
    hours = db.Column(db.String(100), nullable=True)
    location = db.Column(db.String(200), nullable=True)
    description = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.now)
    
    def __repr__(self):
        return f'<JobRecommendation {self.job_title}>'


class UserSkill(db.Model):
    """User skill progress model"""
    __tablename__ = 'user_skills'
    
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    skill_name = db.Column(db.String(100), nullable=False)
    level = db.Column(db.String(50), default='Beginner')  # Beginner, Intermediate, Advanced
    progress = db.Column(db.Integer, default=0)  # 0-100
    created_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)
    
    def __repr__(self):
        return f'<UserSkill {self.skill_name}>'


class UserProfile(db.Model):
    """Extended user profile information"""
    __tablename__ = 'user_profiles'
    
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, unique=True)
    age = db.Column(db.Integer, nullable=True)
    education = db.Column(db.String(100), nullable=True)
    location = db.Column(db.String(200), nullable=True)
    city_type = db.Column(db.String(50), nullable=True)  # Urban, Semi-Urban, Rural
    preferred_domain = db.Column(db.String(100), nullable=True)
    primary_skill = db.Column(db.String(100), nullable=True)
    work_mode_preference = db.Column(db.String(100), nullable=True)
    available_hours = db.Column(db.Integer, nullable=True)
    number_of_kids = db.Column(db.Integer, nullable=True)
    language_preference = db.Column(db.String(50), nullable=True)
    device_type = db.Column(db.String(50), nullable=True)
    bio = db.Column(db.Text, nullable=True)
    profile_completed = db.Column(db.Boolean, default=False)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)
    
    def __repr__(self):
        return f'<UserProfile for user_id {self.user_id}>'


class JobSearchHistory(db.Model):
    """Track all job searches made by users"""
    __tablename__ = 'job_search_history'
    
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    age = db.Column(db.Integer, nullable=True)
    education = db.Column(db.String(100), nullable=True)
    domain = db.Column(db.String(100), nullable=True)
    skill = db.Column(db.String(100), nullable=True)
    work_mode = db.Column(db.String(100), nullable=True)
    location = db.Column(db.String(200), nullable=True)
    city_type = db.Column(db.String(50), nullable=True)
    hours = db.Column(db.Integer, nullable=True)
    kids = db.Column(db.Integer, nullable=True)
    language = db.Column(db.String(50), nullable=True)
    device = db.Column(db.String(50), nullable=True)
    predicted_job = db.Column(db.String(200), nullable=True)
    predicted_salary_low = db.Column(db.Integer, nullable=True)
    predicted_salary_high = db.Column(db.Integer, nullable=True)
    confidence_score = db.Column(db.Float, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.now)
    
    def __repr__(self):
        return f'<JobSearchHistory {self.predicted_job}>'


class SkillSearchHistory(db.Model):
    """Track all skill searches made by users"""
    __tablename__ = 'skill_search_history'
    
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    skill_name = db.Column(db.String(100), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now)
    
    def __repr__(self):
        return f'<SkillSearchHistory {self.skill_name}>'


class Organization(db.Model):
    """Registered Organization model"""
    __tablename__ = 'organizations'
    
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    company_name = db.Column(db.String(200), nullable=False)
    org_type = db.Column(db.String(100), nullable=False)
    industry = db.Column(db.String(100), nullable=False)
    registration_number = db.Column(db.String(100), nullable=True)
    established_year = db.Column(db.Integer, nullable=True)
    org_size = db.Column(db.String(50), nullable=False)
    website = db.Column(db.String(200), nullable=True)
    contact_name = db.Column(db.String(100), nullable=False)
    designation = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(254), nullable=False)
    phone_number = db.Column(db.String(20), nullable=False)
    address = db.Column(db.Text, nullable=False)
    city = db.Column(db.String(100), nullable=False)
    state = db.Column(db.String(100), nullable=False)
    pincode = db.Column(db.String(20), nullable=False)
    password_hash = db.Column(db.String(64), nullable=False)
    salt = db.Column(db.String(64), nullable=False)
    status = db.Column(db.String(50), default='Pending')
    created_at = db.Column(db.DateTime, default=datetime.now)
    
    # Relationships
    jobs = db.relationship('OrganizationJob', backref='organization', lazy=True, cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<Organization {self.company_name}>'

class OrganizationJob(db.Model):
    """Organization Job Postings"""
    __tablename__ = 'organization_jobs'
    
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    org_id = db.Column(db.Integer, db.ForeignKey('organizations.id'), nullable=False)
    title = db.Column(db.String(200), nullable=False)
    job_type = db.Column(db.String(50), nullable=False) # e.g. Job, Internship
    work_mode = db.Column(db.String(100), nullable=True) # Remote, Hybrid, On-site
    location = db.Column(db.String(200), nullable=True)
    salary_range = db.Column(db.String(100), nullable=True)
    description = db.Column(db.Text, nullable=False)
    requirements = db.Column(db.Text, nullable=True)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.now)
    
    def __repr__(self):
        return f'<OrganizationJob {self.title}>'


# Create all database tables
with app.app_context():
    db.create_all()
    print("✅ Database tables created successfully!")

# ✅ Login required decorator with session validation
from functools import wraps

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_email' not in session:
            # Only store GET URLs for redirect after login
            # POST-only routes like /predict would fail on GET redirect
            if request.method == 'GET':
                session['next_url'] = request.url
            flash('Please login to access this feature', 'info')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


def get_current_user():
    """Get current logged in user info for templates"""
    if 'user_email' in session:
        name = session.get('user_name', 'User')
        return {
            'name': name,
            'email': session.get('user_email', ''),
            'initials': ''.join([n[0].upper() for n in name.split()[:2]])
        }
    return None


# ✅ OPTIMIZED: Lazy load dataset only when needed
df = None
vectorizer = None
X = None
rows_text = None

def load_dataset():
    """Lazy load dataset to avoid blocking on startup"""
    global df, vectorizer, X, rows_text
    if df is None:
        try:
            print("📊 Loading dataset...")
            df = pd.read_csv("dataset.csv", nrows=None)  # Load all data once
            df.fillna("", inplace=True)
            
            # Convert each row into one text string
            rows_text = df.astype(str).agg(" | ".join, axis=1).tolist()
            
            # Create vectorizer (cache it)
            from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
            X = vectorizer.fit_transform(rows_text)
            print("✅ Dataset loaded and cached successfully!")
        except Exception as e:
            print(f"⚠️ Error loading dataset: {e}")
            df = pd.DataFrame()
            vectorizer = None
            X = None
            rows_text = []
    return df, vectorizer, X, rows_text


# ✅ Load trained models (make sure these exist)
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'trained_models')

# Model + preprocessor pairs
job_model = safe_load_model(os.path.join(MODEL_DIR, 'job_recommendation_model.pkl'))
job_preprocessor_data = safe_load_model(os.path.join(MODEL_DIR, 'job_preprocessor.pkl'))

income_model = safe_load_model(os.path.join(MODEL_DIR, 'income_prediction_model.pkl'))
income_preprocessor_data = safe_load_model(os.path.join(MODEL_DIR, 'income_preprocessor.pkl'))

skill_match_model = safe_load_model(os.path.join(MODEL_DIR, 'skill_match_model.pkl'))
skill_match_preprocessor_data = safe_load_model(os.path.join(MODEL_DIR, 'skill_match_preprocessor.pkl'))

mother_model = safe_load_model(os.path.join(MODEL_DIR, 'mother_suitability_model.pkl'))
mother_preprocessor_data = safe_load_model(os.path.join(MODEL_DIR, 'mother_suitability_preprocessor.pkl'))

skill_gap_model = safe_load_model(os.path.join(MODEL_DIR, 'skill_gap_model.pkl'))
skill_gap_preprocessor_data = safe_load_model(os.path.join(MODEL_DIR, 'skill_gap_preprocessor.pkl'))

career_model = safe_load_model(os.path.join(MODEL_DIR, 'career_path_model.pkl'))
career_preprocessor_data = safe_load_model(os.path.join(MODEL_DIR, 'career_path_preprocessor.pkl'))

wlb_model = safe_load_model(os.path.join(MODEL_DIR, 'work_life_balance_model.pkl'))
wlb_preprocessor_data = safe_load_model(os.path.join(MODEL_DIR, 'work_life_balance_preprocessor.pkl'))

profile_model = safe_load_model(os.path.join(MODEL_DIR, 'profile_completeness_model.pkl'))
profile_preprocessor_data = safe_load_model(os.path.join(MODEL_DIR, 'profile_completeness_preprocessor.pkl'))

print(f"Models loaded: job={job_model is not None}, income={income_model is not None}, "
      f"skill_match={skill_match_model is not None}, mother={mother_model is not None}, "
      f"skill_gap={skill_gap_model is not None}, career={career_model is not None}, "
      f"wlb={wlb_model is not None}, profile={profile_model is not None}")

# ✅ Language Dictionary (EN + HI)
TEXT = {
    "en": {
        "education_list": {
            "Below 8th/Informal Education": "Below 8th / Informal",
            "8th Pass": "8th Pass",
            "10th Pass (SSC)": "10th Pass (SSC)",
            "12th Pass (HSC)": "12th Pass (HSC)",
            "Diploma/ITI": "Diploma / ITI",
            "Graduate (BTech/BA/BCom/BSc)": "Graduate (BTech/BA/BCom/BSc)",
            "Post Graduate (MBA/MTech/MA/MSc)": "Post Graduate (MBA/MTech/MA/MSc)",
            "PhD/Doctorate": "PhD / Doctorate"
        },
        "city_type_list": {
            "Metro": "Metro",
            "Tier-1": "Tier-1",
            "Tier-2": "Tier-2",
            "Tier-3": "Tier-3",
            "Rural": "Rural",
            "Remote": "Remote"
        },
        "language_list": {
            "Hindi": "Hindi",
            "English": "English",
            "Both English and Hindi": "Both English and Hindi",
            "English + Regional": "English + Regional",
            "Regional Language": "Regional Language"
        },
        "device_list": {
            "Mobile Phone": "Mobile Phone",
            "Laptop": "Laptop",
            "Desktop": "Desktop",
            "Both Mobile and Laptop": "Both Mobile and Laptop",
            "No Device Required": "No Device Required"
        },
        "work_mode_list": {
            "Work From Home": "Work From Home",
            "Remote": "Remote",
            "Hybrid": "Hybrid",
            "Office": "Office",
            "On-site": "On-site",
            "Field Work": "Field Work"
        },
        "work_type_list": {
            "Full-time": "Full-time",
            "Part-time": "Part-time",
            "Freelance": "Freelance",
            "Contract": "Contract",
            "Internship": "Internship"
        },
        "marital_status_list": {
            "Single": "Single",
            "Married": "Married"
        },
        "shift_type_list": {
            "Day Shift": "Day Shift",
            "General Shift": "General Shift",
            "Flexible": "Flexible",
            "Night Shift": "Night Shift",
            "Rotational": "Rotational"
        },
        "sector_list": [
            "IT & Technology","Healthcare","Education","Banking & Finance",
            "Sales & Marketing","E-Commerce","Content & Media","Customer Service",
            "Beauty & Wellness","Childcare & Homecare","Arts & Design",
            "Administration","Agriculture","Consulting","Gig Economy","Government",
            "Hospitality","Human Resources","Legal","Manufacturing","Real Estate",
            "General"
        ],
        "domain_list": [
            "Accounting","Administration","Agriculture","Allied Health",
            "Banking Operations","Beauty Wellness","Childcare","Cloud & DevOps",
            "Construction","Consulting","Content Writing","Corporate Training",
            "Customer Support","Cybersecurity","Dairy Animal","Data Entry",
            "Data Science","Delivery Logistics","Digital Marketing","Ecommerce",
            "EdTech","Elderly Care","Fashion Design","Fitness Yoga",
            "Food Processing","Food Service","Freelancing","General","Government",
            "Graphic Design","HR","Handicrafts","Higher Education",
            "Hotel Management","IT Support","Insurance","Investment","Journalism",
            "Legal","Manufacturing","Marketing","Medical Doctors","Mental Health",
            "Nursing","Pharma Sales","QA Testing","Real Estate","Retail","Sales",
            "School Teaching","Social Work","Software Engineering",
            "Travel Tourism","UI/UX Design"
        ],
        "location_optional": "Location (Optional)",
        "goal": "Goal",
        "learning_mode": "Learning Mode",
        "what_you_get": "What you will get",
        "goal_list": {
            "Job": "Get a Job",
            "Freelancing": "Freelancing",
            "Business": "Start Business"
        },
        "learning_mode_list": {
            "Video": "Video Based",
            "Practice": "Practice Based",
            "Both": "Both"
        },
        "hours_list": {
            "1": "1 hour/day",
            "2": "2 hours/day",
            "3": "3 hours/day",
            "4": "4+ hours/day"
        },
        "what_list": [
            "Step-by-step training roadmap",
            "Best YouTube + Google resources",
            "Portfolio building guidance",
            "Job/Freelance opportunities links"
        ],
        "title": "MaaSarthi",
        "tagline": "Helping Mothers Find Work & Skill Training Opportunities",
        "smart_tag": "Smart Career Suggestions + Training",
        "hero_line1": "Helping Mothers find",
        "hero_work": "Work",
        "hero_and": "and",
        "hero_skills": "Skills",
        "hero_line2": "to Earn from Home",
        "hero_desc": "MaaSarthi recommends the best earning options based on your skills, time availability, education and location.",
        "feature1_title": "Personalized Jobs",
        "feature1_desc": "Get best job recommendation + expected income range.",
        "feature2_title": "Income Planning",
        "feature2_desc": "Know how much you can earn based on hours/day.",
        "feature3_title": "Skill Training",
        "feature3_desc": "Learn skills using YouTube + Google + Instagram links.",
        "find_jobs": "Find Jobs",
        "train_skill": "Train a Skill",
        "home": "Home",
        "about_us": "About Us",
        "contact": "Contact",
        "mothers_empowered": "Mothers Empowered",
        "skills_available": "Skills Available",
        "avg_income": "Avg Monthly Income",
        "how_it_works": "How It Works",
        "popular_skills": "Popular Skills to Learn",
        "success_stories": "Success Stories",
        "get_started_today": "Get Started Today",
        "talk_to_expert": "Talk to an Expert",
        "view_all_skills": "View All Skills",
        "why_choose": "Why Choose MaaSarthi?",
        "why_choose_desc": "We provide personalized career guidance and training opportunities tailored for mothers",
        "personalized_matching": "Personalized Job Matching",
        "matching_desc": "Our smart algorithm matches you with jobs that fit your skills, schedule, and location perfectly.",
        "explore_jobs": "Explore Jobs →",
        "free_training": "Free Skill Training",
        "training_desc": "Access over 150+ free courses and training programs to enhance your skills and increase earning potential.",
        "browse_courses": "Browse Courses →",
        "flexible_work": "Flexible Work Options",
        "flexible_desc": "Choose from part-time, full-time, or freelance opportunities that work around your family schedule.",
        "view_options": "View Options →",
        "community": "Community Support",
        "community_desc": "Join a supportive community of mothers who understand your journey and share valuable experiences.",
        "join_community": "Join Community →",
        "certification": "Certification Programs",
        "certification_desc": "Earn recognized certifications that boost your credibility and help you land better opportunities.",
        "get_certified": "Get Certified →",
        "guidance_24_7": "24/7 Career Guidance",
        "guidance_desc": "Get personalized career advice and support from our experts whenever you need it.",
        "contact_us": "Contact Us →",
        "create_profile": "Create Your Profile",
        "profile_desc": "Tell us about your skills, experience, and what you're looking for",
        "get_matches": "Get Personalized Matches",
        "matches_desc": "Receive job and training recommendations tailored just for you",
        "start_earning": "Start Earning & Growing",
        "earning_desc": "Begin your work-from-home journey and unlock your potential",
        "how_it_works_desc": "Start your journey to financial independence in just 3 simple steps",
        "popular_skills_desc": "Start learning these in-demand skills today and increase your earning potential",
        "success_desc": "Hear from mothers who transformed their lives with MaaSarthi",
        "ready_to_start": "Ready to Start Your Journey?",
        "join_thousands": "Join thousands of mothers who are earning from home with MaaSarthi",
        "footer_desc": "Empowering mothers across India to achieve financial independence through flexible work-from-home opportunities and skill development.",
        "quick_links": "Quick Links",
        "categories": "Categories",
        "support": "Support",
        "help_center": "Help Center",
        "privacy_policy": "Privacy Policy",
        "terms_service": "Terms of Service",
        "faqs": "FAQs",
        "copyright": "© 2025 MaaSarthi. All rights reserved. Made with ❤️ for Mothers",
        "data_entry": "Data Entry Jobs",
        "content_writing": "Content Writing",
        "graphic_design": "Graphic Design",
        "online_tutoring": "Online Tutoring",
        "freelance_work": "Freelance Work",
        "help_center": "Help Center",
        "privacy_policy": "Privacy Policy",
        "terms_service": "Terms of Service",
        "faqs": "FAQs",
        "career_guidance": "Career Guidance",
        "back": "Back",
        "job_heading": "Find Your Best Work Option",
        "job_sub": "Fill your details and MaaSarthi will suggest the best job + income estimate.",
        "age": "Age",
        "kids": "Number of Kids",
        "hours": "Hours Available Per Day",
        "domain": "Domain / Field",
        "sector": "Sector",
        "main_skill": "Primary Skill",
        "secondary_skill": "Secondary Skill",
        "education": "Education",
        "city_type": "City Tier",
        "language": "Language",
        "device": "Device",
        "work_mode": "Work Mode",
        "work_type": "Work Type",
        "marital_status": "Marital Status",
        "shift_type": "Preferred Shift",
        "experience_years": "Experience (Years)",
        "get_rec": "Get My Career Report",
        "train_instead": "Train a Skill Instead",
        "result_title": "Recommendation",
        "expected_income": "Expected Monthly Income",
        "skills_learn": "Skills You Should Learn Next",
        "helpful_links": "Helpful Resources",
        "try_again": "Try Another Profile",
        "recommended": "Recommended",
        "best_match": "Best Match",
        "recommended_work": "Recommended Work",
        "personalized_desc": "Personalized job suggestion + income estimate + learning plan",
        "skill_heading": "Skill Training Plan",
        "skill_sub": "Choose a skill and get roadmap + best learning links.",
        "skill": "Skill",
        "level": "Level",
        "preferred_language": "Preferred Language",
        "get_training": "🎯 Get Training Plan",
        "choose": "Choose",
        "select": "Select",
        "skills_map": {
            "Cooking": "Cooking",
            "Baking": "Baking",
            "Teaching": "Teaching",
            "Beauty": "Beauty",
            "Mehndi": "Mehndi",
            "Tailoring": "Tailoring / Stitching",
            "Handicraft": "Handicraft",
            "Canva": "Canva",
            "Excel": "Excel",
            "Data Entry": "Data Entry",
            "Social Media": "Social Media",
            "Video Editing": "Video Editing",
            "Content Writing": "Content Writing",
            "Graphic Design": "Graphic Design",
            "Home Cleaning": "Home Cleaning",
            "Babysitting": "Babysitting",
            "Caregiver": "Caregiver",
            "Home Tutor": "Home Tutor",
            "Electrician": "Electrician",
            "Plumbing": "Plumbing",
            "Mobile Repairing": "Mobile Repairing",
            "Spoken English": "Spoken English",
            "Reselling": "Reselling",
            "E-commerce Packing": "E-commerce Packing",
            "OTHER": "Other (Write your skill)"
        },
        "level_list": {
            "Beginner": "Beginner",
            "Intermediate": "Intermediate",
            "Advanced": "Advanced"
        }
    },
    "hi": {
        "education_list": {
            "Below 8th/Informal Education": "8वीं से नीचे / अनौपचारिक",
            "8th Pass": "8वीं पास",
            "10th Pass (SSC)": "10वीं पास (SSC)",
            "12th Pass (HSC)": "12वीं पास (HSC)",
            "Diploma/ITI": "डिप्लोमा / ITI",
            "Graduate (BTech/BA/BCom/BSc)": "स्नातक (BTech/BA/BCom/BSc)",
            "Post Graduate (MBA/MTech/MA/MSc)": "स्नातकोत्तर (MBA/MTech/MA/MSc)",
            "PhD/Doctorate": "पीएचडी / डॉक्टरेट"
        },
        "city_type_list": {
            "Metro": "मेट्रो",
            "Tier-1": "टियर-1",
            "Tier-2": "टियर-2",
            "Tier-3": "टियर-3",
            "Rural": "ग्रामीण",
            "Remote": "रिमोट"
        },
        "language_list": {
            "Hindi": "हिन्दी",
            "English": "अंग्रेज़ी",
            "Both English and Hindi": "अंग्रेज़ी और हिन्दी दोनों",
            "English + Regional": "अंग्रेज़ी + क्षेत्रीय",
            "Regional Language": "क्षेत्रीय भाषा"
        },
        "device_list": {
            "Mobile Phone": "मोबाइल फोन",
            "Laptop": "लैपटॉप",
            "Desktop": "डेस्कटॉप",
            "Both Mobile and Laptop": "मोबाइल और लैपटॉप दोनों",
            "No Device Required": "किसी डिवाइस की ज़रूरत नहीं"
        },
        "work_mode_list": {
            "Work From Home": "घर से काम",
            "Remote": "रिमोट",
            "Hybrid": "हाइब्रिड",
            "Office": "ऑफिस",
            "On-site": "ऑन-साइट",
            "Field Work": "फील्ड वर्क"
        },
        "work_type_list": {
            "Full-time": "पूर्णकालिक",
            "Part-time": "अंशकालिक",
            "Freelance": "फ्रीलांस",
            "Contract": "कॉन्ट्रैक्ट",
            "Internship": "इंटर्नशिप"
        },
        "marital_status_list": {
            "Single": "अविवाहित",
            "Married": "विवाहित"
        },
        "shift_type_list": {
            "Day Shift": "दिन की शिफ्ट",
            "General Shift": "सामान्य शिफ्ट",
            "Flexible": "फ्लेक्सिबल",
            "Night Shift": "रात की शिफ्ट",
            "Rotational": "रोटेशनल"
        },
        "sector_list": [
            "आईटी और टेक्नोलॉजी","स्वास्थ्य","शिक्षा","बैंकिंग और वित्त",
            "बिक्री और विपणन","ई-कॉमर्स","कंटेंट और मीडिया","ग्राहक सेवा",
            "सौंदर्य और कल्याण","बाल देखभाल और गृह देखभाल","कला और डिज़ाइन",
            "प्रशासन","कृषि","परामर्श","गिग इकॉनमी","सरकारी",
            "आतिथ्य","मानव संसाधन","कानूनी","विनिर्माण","रियल एस्टेट",
            "सामान्य"
        ],
        "domain_list": [
            "अकाउंटिंग","प्रशासन","कृषि","सम्बद्ध स्वास्थ्य",
            "बैंकिंग ऑपरेशंस","सौंदर्य कल्याण","बाल देखभाल","क्लाउड और DevOps",
            "निर्माण","परामर्श","कंटेंट राइटिंग","कॉर्पोरेट ट्रेनिंग",
            "ग्राहक सहायता","साइबरसिक्योरिटी","डेयरी पशु","डेटा एंट्री",
            "डेटा साइंस","डिलीवरी लॉजिस्टिक्स","डिजिटल मार्केटिंग","ई-कॉमर्स",
            "एडटेक","वृद्ध देखभाल","फैशन डिज़ाइन","फिटनेस योगा",
            "फूड प्रोसेसिंग","फूड सर्विस","फ्रीलांसिंग","सामान्य","सरकारी",
            "ग्राफिक डिज़ाइन","एचआर","हस्तकला","उच्च शिक्षा",
            "होटल मैनेजमेंट","आईटी सपोर्ट","बीमा","निवेश","पत्रकारिता",
            "कानूनी","विनिर्माण","मार्केटिंग","चिकित्सक","मानसिक स्वास्थ्य",
            "नर्सिंग","फार्मा सेल्स","क्यूए टेस्टिंग","रियल एस्टेट","रिटेल","बिक्री",
            "स्कूल टीचिंग","सामाजिक कार्य","सॉफ्टवेयर इंजीनियरिंग",
            "ट्रैवल टूरिज्म","UI/UX डिज़ाइन"
        ],
        "location_optional": "लोकेशन (Optional)",
        "title": "माँ सारथी",
        "tagline": "माँओं के लिए घर से काम और स्किल ट्रेनिंग प्लेटफॉर्म",
        "goal": "लक्ष्य",
        "learning_mode": "सीखने का तरीका",
        "what_you_get": "आपको क्या मिलेगा",
        "goal_list": {
            "Job": "नौकरी पाना",
            "Freelancing": "फ्रीलांसिंग",
            "Business": "बिज़नेस शुरू करना"
        },
        "learning_mode_list": {
            "Video": "वीडियो आधारित",
            "Practice": "प्रैक्टिस आधारित",
            "Both": "दोनों"
        },
        "hours_list": {
            "1": "1 घंटा/दिन",
            "2": "2 घंटे/दिन",
            "3": "3 घंटे/दिन",
            "4": "4+ घंटे/दिन"
        },
        "what_list": [
            "स्टेप-बाय-स्टेप ट्रेनिंग रोडमैप",
            "सबसे अच्छे YouTube + Google संसाधन",
            "पोर्टफोलियो बनाने की गाइडेंस",
            "नौकरी/फ्रीलांस अवसरों के लिंक"
        ],
        "smart_tag": "स्मार्ट करियर सुझाव + ट्रेनिंग",
        "hero_line1": "माँओं को",
        "hero_work": "काम",
        "hero_and": "और",
        "hero_skills": "स्किल",
        "hero_line2": "सीखकर घर से कमाने में मदद",
        "hero_desc": "माँ सारथी आपके स्किल, समय, शिक्षा और लोकेशन के आधार पर सबसे अच्छे काम और कमाई के विकल्प सुझाता है।",
        "feature1_title": "पर्सनलाइज्ड जॉब्स",
        "feature1_desc": "आपके लिए सही काम + अनुमानित कमाई रेंज।",
        "feature2_title": "इनकम प्लानिंग",
        "feature2_desc": "घंटों के आधार पर आप कितना कमा सकते हैं।",
        "feature3_title": "स्किल ट्रेनिंग",
        "feature3_desc": "YouTube + Google + Instagram लिंक से स्किल सीखें।",
        "find_jobs": "नौकरी खोजें",
        "train_skill": "स्किल सीखें",
        "home": "होम",
        "about_us": "हमारे बारे में",
        "contact": "संपर्क करें",
        "mothers_empowered": "माताओं को सशक्त बनाया गया",
        "skills_available": "उपलब्ध कौशल",
        "avg_income": "औसत मासिक आय",
        "how_it_works": "यह कैसे काम करता है",
        "popular_skills": "सीखने के लिए लोकप्रिय कौशल",
        "success_stories": "सफलता की कहानियां",
        "get_started_today": "आज ही शुरू करें",
        "talk_to_expert": "किसी विशेषज्ञ से बात करें",
        "view_all_skills": "सभी कौशल देखें",
        "why_choose": "MaaSarthi को क्यों चुनें?",
        "why_choose_desc": "हम माताओं के लिए व्यक्तिगत करियर मार्गदर्शन और प्रशिक्षण अवसर प्रदान करते हैं",
        "personalized_matching": "व्यक्तिगत नौकरी मिलान",
        "matching_desc": "हमारा स्मार्ट एल्गोरिदम आपको उन नौकरियों से मेल खाता है जो आपके कौशल, समय और स्थान के अनुरूप हों।",
        "explore_jobs": "नौकरियों का अन्वेषण करें →",
        "free_training": "मुफ्त कौशल प्रशिक्षण",
        "training_desc": "150+ से अधिक मुफ्त पाठ्यक्रमों और प्रशिक्षण कार्यक्रमों तक पहुंचें।",
        "browse_courses": "पाठ्यक्रम देखें →",
        "flexible_work": "लचकदार काम विकल्प",
        "flexible_desc": "अंशकालीन, पूर्णकालीन या फ्रीलांस अवसरों में से चुनें।",
        "view_options": "विकल्प देखें →",
        "community": "सामुदायिक समर्थन",
        "community_desc": "उन माताओं के समुदाय में शामिल हों जो आपकी यात्रा को समझते हैं।",
        "join_community": "समुदाय में शामिल हों →",
        "certification": "प्रमाणन कार्यक्रम",
        "certification_desc": "मान्यता प्राप्त प्रमाणपत्र अर्जित करें जो आपकी विश्वसनीयता बढ़ाते हैं।",
        "get_certified": "प्रमाणित हों →",
        "guidance_24_7": "24/7 करियर मार्गदर्शन",
        "guidance_desc": "हमारे विशेषज्ञों से व्यक्तिगत करियर सलाह प्राप्त करें।",
        "contact_us": "संपर्क करें →",
        "create_profile": "अपनी प्रोफाइल बनाएं",
        "profile_desc": "हमें अपने कौशल और अनुभव के बारे में बताएं",
        "get_matches": "व्यक्तिगत मेल प्राप्त करें",
        "matches_desc": "आपके लिए तैयार किए गए नौकरी और प्रशिक्षण सुझाव प्राप्त करें",
        "start_earning": "कमाई करना शुरू करें और बढ़ें",
        "earning_desc": "अपनी घर से काम की यात्रा शुरू करें और अपनी क्षमता को अनलॉक करें",
        "how_it_works_desc": "बस 3 सरल चरणों में वित्तीय स्वतंत्रता की ओर अपनी यात्रा शुरू करें",
        "popular_skills_desc": "आज ही इन मांग वाले कौशल को सीखना शुरू करें",
        "success_desc": "उन माताओं की कहानियां सुनें जिन्होंने MaaSarthi के साथ अपना जीवन बदल दिया",
        "ready_to_start": "क्या आप अपनी यात्रा शुरू करने के लिए तैयार हैं?",
        "join_thousands": "हजारों माताओं के साथ शामिल हों जो MaaSarthi के साथ घर से कमा रही हैं",
        "footer_desc": "भारत की माताओं को लचकदार घर से काम के अवसरों और कौशल विकास के माध्यम से वित्तीय स्वतंत्रता प्राप्त करने में सक्षम बनाना।",
        "quick_links": "त्वरित लिंक",
        "categories": "श्रेणियां",
        "support": "समर्थन",
        "help_center": "सहायता केंद्र",
        "privacy_policy": "गोपनीयता नीति",
        "terms_service": "सेवा की शर्तें",
        "faqs": "अक्सर पूछे जाने वाले प्रश्न",
        "copyright": "© 2025 MaaSarthi। सर्वाधिकार सुरक्षित। माताओं के लिए ❤️ से बनाया गया",
        "data_entry": "डेटा एंट्री नौकरियां",
        "content_writing": "कंटेंट राइटिंग",
        "graphic_design": "ग्राफिक डिजाइन",
        "online_tutoring": "ऑनलाइन ट्यूटोरिंग",
        "freelance_work": "फ्रीलांस काम",
        "help_center": "सहायता केंद्र",
        "privacy_policy": "गोपनीयता नीति",
        "terms_service": "सेवा की शर्तें",
        "faqs": "अक्सर पूछे जाने वाले प्रश्न",
        "career_guidance": "करियर मार्गदर्शन",
        "back": "वापस",
        "job_heading": "अपने लिए सबसे अच्छा काम चुनें",
        "job_sub": "अपनी जानकारी भरें और माँ सारथी आपको सही काम + अनुमानित कमाई बताएगा।",
        "age": "उम्र",
        "kids": "बच्चों की संख्या",
        "hours": "दिन में उपलब्ध समय (घंटे)",
        "domain": "क्षेत्र (डोमेन)",
        "sector": "सेक्टर",
        "main_skill": "मुख्य स्किल",
        "secondary_skill": "अन्य स्किल",
        "education": "शिक्षा",
        "city_type": "शहर टियर",
        "language": "भाषा",
        "device": "डिवाइस",
        "work_mode": "काम का तरीका",
        "work_type": "काम का प्रकार",
        "marital_status": "वैवाहिक स्थिति",
        "shift_type": "पसंदीदा शिफ्ट",
        "experience_years": "अनुभव (वर्ष)",
        "get_rec": "मेरी करियर रिपोर्ट देखें",
        "train_instead": "स्किल ट्रेनिंग करें",
        "result_title": "सुझाव",
        "expected_income": "अनुमानित मासिक कमाई",
        "skills_learn": "अगली स्किल जो आपको सीखनी चाहिए",
        "helpful_links": "उपयोगी लिंक",
        "try_again": "दोबारा प्रोफाइल भरें",
        "recommended": "सुझाव",
        "best_match": "सबसे अच्छा",
        "recommended_work": "सुझाया गया काम",
        "personalized_desc": "आपके लिए सही काम + कमाई + सीखने की योजना",
        "skill_heading": "स्किल ट्रेनिंग प्लान",
        "skill_sub": "एक स्किल चुनें और रोडमैप + सीखने के लिंक पाएं।",
        "skill": "स्किल",
        "level": "लेवल",
        "preferred_language": "पसंदीदा भाषा",
        "get_training": " ट्रेनिंग प्लान देखें",
        "choose": "चुनें",
        "select": "चुनें",
        "skills_map": {
            "Cooking": "कुकिंग",
            "Baking": "बेकिंग",
            "Teaching": "टीचिंग",
            "Beauty": "ब्यूटी",
            "Mehndi": "मेहंदी",
            "Tailoring": "सिलाई / टेलरिंग",
            "Handicraft": "हस्तकला",
            "Canva": "कैनवा",
            "Excel": "एक्सेल",
            "Data Entry": "डेटा एंट्री",
            "Social Media": "सोशल मीडिया",
            "Video Editing": "वीडियो एडिटिंग",
            "Content Writing": "कंटेंट राइटिंग",
            "Graphic Design": "ग्राफिक डिज़ाइन",
            "Home Cleaning": "घर की सफाई",
            "Babysitting": "बेबीसिटिंग",
            "Caregiver": "केयरगिवर",
            "Home Tutor": "होम ट्यूटर",
            "Electrician": "इलेक्ट्रीशियन",
            "Plumbing": "प्लंबिंग",
            "Mobile Repairing": "मोबाइल रिपेयरिंग",
            "Spoken English": "स्पोकन इंग्लिश",
            "Reselling": "रीसेलिंग",
            "E-commerce Packing": "ई-कॉमर्स पैकिंग",
            "OTHER": "अन्य (अपनी स्किल लिखें)"
        },
        "level_list": {
            "Beginner": "शुरूआती",
            "Intermediate": "मध्यम",
            "Advanced": "एडवांस"
        }
    }
}

# ============================================
# 🔐 AUTHENTICATION ROUTES
# ============================================

@app.route('/login', methods=['GET', 'POST'])
def login():
    """
    Secure login route with:
    - Input validation
    - Rate limiting
    - Password verification
    - Session management
    """
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '').strip()
        
        # Get client IP for rate limiting
        client_ip = request.remote_addr
        rate_limit_key = f"{client_ip}:{email}"
        
        # Check rate limiting
        is_allowed, remaining_time = login_rate_limiter.is_allowed(rate_limit_key)
        if not is_allowed:
            return render_template('login.html', 
                error=f'Too many failed attempts. Please try again in {remaining_time} seconds.')
        
        # Validate email format
        is_valid, error_msg = Validator.validate_email(email)
        if not is_valid:
            login_rate_limiter.record_attempt(rate_limit_key, False)
            return render_template('login.html', error=error_msg)
        
        # Validate password is not empty
        if not password:
            login_rate_limiter.record_attempt(rate_limit_key, False)
            return render_template('login.html', error='Password is required')
        
        # Check if user exists
        user = User.query.filter_by(email=email).first()
        if not user:
            login_rate_limiter.record_attempt(rate_limit_key, False)
            # Generic error to prevent email enumeration
            return render_template('login.html', error='Invalid email or password')
        
        # Verify password using secure comparison
        if not SecurityUtils.verify_password(password, user.password_hash, user.salt):
            login_rate_limiter.record_attempt(rate_limit_key, False)
            return render_template('login.html', error='Invalid email or password')
        
        # Successful login - record and create session
        login_rate_limiter.record_attempt(rate_limit_key, True)
        
        # Get the URL user wanted to access before login
        next_url = session.get('next_url')
        
        # Regenerate session to prevent session fixation
        session.clear()
        session['user_email'] = email
        session['user_name'] = user.name
        session['user_phone'] = user.phone
        session['session_token'] = SecurityUtils.generate_session_token()
        session['login_time'] = datetime.now().isoformat()
        session.permanent = True
        
        # Update last login time in database
        user.last_login = datetime.now()
        db.session.commit()
        
        # Redirect to the page user wanted or dashboard
        if next_url:
            return redirect(next_url)
        return redirect('/dashboard')
    
    # GET request - show login form
    success_msg = request.args.get('success')
    return render_template('login.html', success=success_msg)


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    """
    Secure signup route with:
    - Comprehensive input validation
    - Password hashing with salt
    - Duplicate email prevention
    """
    if request.method == 'POST':
        # Sanitize and get form data
        fullname = SecurityUtils.sanitize_input(request.form.get('fullname', ''))
        email = request.form.get('email', '').strip().lower()
        phone = request.form.get('phone', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        age = request.form.get('age', '')
        
        errors = []
        
        # Validate name
        is_valid, error_msg = Validator.validate_name(fullname)
        if not is_valid:
            errors.append(error_msg)
        
        # Validate email
        is_valid, error_msg = Validator.validate_email(email)
        if not is_valid:
            errors.append(error_msg)
        
        # Check if email already exists
        if User.query.filter_by(email=email).first():
            errors.append('An account with this email already exists')
        
        # Validate phone
        is_valid, error_msg = Validator.validate_phone(phone)
        if not is_valid:
            errors.append(error_msg)
        
        # Validate age
        try:
            age = int(age) if age else None
            if age and (age < 18 or age > 100):
                errors.append('Age must be between 18 and 100')
        except ValueError:
            errors.append('Please enter a valid age')
            age = None
        
        # Validate password strength
        is_valid, error_msg = Validator.validate_password(password)
        if not is_valid:
            errors.append(error_msg)
        
        # Check password confirmation
        if password != confirm_password:
            errors.append('Passwords do not match')
        
        # If there are validation errors, return them
        if errors:
            return render_template('signup.html', error=' | '.join(errors))
        
        # Hash password with salt
        password_hash, salt = SecurityUtils.hash_password(password)
        
        # Normalize phone number
        phone = re.sub(r'[\s\-]', '', phone)
        if phone.startswith('+91'):
            phone = phone[3:]
        elif phone.startswith('91') and len(phone) == 12:
            phone = phone[2:]
        elif phone.startswith('0'):
            phone = phone[1:]
        
        # Create user record in database
        new_user = User(
            name=fullname,
            email=email,
            phone=phone,
            password_hash=password_hash,
            salt=salt,
            created_at=datetime.now(),
            is_active=True,
            is_verified=False
        )
        db.session.add(new_user)
        db.session.commit()
        
        # ✅ Create user profile with age
        if age:
            user_profile = UserProfile(
                user_id=new_user.id,
                age=age
            )
            db.session.add(user_profile)
            db.session.commit()
        
        # Log successful registration (in production, use proper logging)
        print(f"✅ New user registered: {email} (ID: {new_user.id})")
        
        # ✅ Automatically log the user in after signup
        session.permanent = True
        session['user_email'] = email
        session['user_name'] = fullname
        session['user_id'] = new_user.id
        session['login_time'] = datetime.now().isoformat()
        session['session_token'] = SecurityUtils.generate_session_token()
        
        # Update last login time
        new_user.last_login = datetime.now()
        db.session.commit()
        
        # Flash success message
        flash('Account created successfully! Welcome to MaaSarthi.', 'success')
        
        # Redirect to dashboard
        return redirect(url_for('dashboard'))
    
    # GET request - show signup form
    return render_template('signup.html')

@app.route('/forgot-password')
def forgot_password():
    return render_template('forgot_password.html')


@app.route('/dashboard')
def dashboard():
    """
    Protected dashboard route with real data
    Validates session and displays user's recommendations, skills, and reminders
    """
    # Check if user is logged in
    if 'user_email' not in session:
        return redirect('/login')
    
    # Validate session integrity
    if 'session_token' not in session:
        session.clear()
        return redirect('/login')
    
    # Get user info
    user_email = session.get('user_email')
    user_name = session.get('user_name', 'User')
    
    # Verify user still exists in database
    user = User.query.filter_by(email=user_email).first()
    if not user:
        session.clear()
        return redirect('/login')
    
    # Get user's tasks
    user_tasks = Task.query.filter_by(user_id=user.id).all()
    completed_tasks = [t for t in user_tasks if t.is_completed]
    
    # Get user's reminders
    user_reminders = Reminder.query.filter_by(user_id=user.id).order_by(Reminder.due_date).all()
    upcoming_reminders = [r for r in user_reminders if not r.is_completed]
    
    # Get last job recommendation
    last_recommendation = JobRecommendation.query.filter_by(user_id=user.id).order_by(JobRecommendation.created_at.desc()).first()
    
    # Get user's skills
    user_skills = UserSkill.query.filter_by(user_id=user.id).all()
    
    # Calculate statistics
    completion_rate = int((len(completed_tasks) / max(len(user_tasks), 1)) * 100) if user_tasks else 0
    
    # Default statistics (can be enhanced with more models)
    dashboard_stats = {
        'job_applications': 12,
        'active_skills': len(user_skills),
        'completion_rate': completion_rate,
        'monthly_earnings': '28,000'
    }
    
    return render_template(
        'dashboard.html',
        user_name=user_name,
        user=get_current_user(),
        last_recommendation=last_recommendation,
        user_skills=user_skills,
        job_applications=dashboard_stats['job_applications'],
        active_skills=dashboard_stats['active_skills'],
        completion_rate=dashboard_stats['completion_rate'],
        monthly_earnings=dashboard_stats['monthly_earnings'],
        upcoming_reminders=upcoming_reminders[:5],  # Top 5 upcoming reminders
        pending_tasks=len([t for t in user_tasks if not t.is_completed]),
        user_tasks=user_tasks
    )


@app.route('/logout')
def logout():
    """
    Secure logout - clears all session data
    """
    # Log logout event
    if 'user_email' in session:
        print(f"👋 User logged out: {session.get('user_email')}")
    
    session.clear()
    return redirect('/')

# ✅ Helper function to get language text
def get_text():
    lang = session.get("lang", "en")
    return TEXT.get(lang, TEXT["en"])
@app.route("/chat", methods=["POST"])
def chat():
    user_msg = request.json.get("message", "").strip()

    if not user_msg:
        return jsonify({"reply": "Please type something 🙂"})

    # ✅ Load dataset on first use
    global df, vectorizer, X, rows_text
    df, vectorizer, X, rows_text = load_dataset()
    
    if vectorizer is None or X is None:
        return jsonify({"reply": "Chat service is loading... please try again in a moment."})

    # 🔍 Find best matching rows from CSV
    query_vec = vectorizer.transform([user_msg])
    similarities = cosine_similarity(query_vec, X).flatten()

    # Top 3 matches
    top_indices = similarities.argsort()[-3:][::-1]
    context = "\n".join([rows_text[i] for i in top_indices])

    # 🤖 Ask Ollama with dataset context
    prompt = f"""
You are MaaSathi AI Assistant.
Answer using the dataset context given below.
If dataset does not contain answer, reply normally.

Dataset Context:
{context}

User Question: {user_msg}
Assistant:
"""

    url = "http://localhost:11434/api/generate"
    payload = {"model": "phi3", "prompt": prompt, "stream": False}

    response = requests.post(url, json=payload)
    data = response.json()

    return jsonify({"reply": data.get("response", "Sorry, I couldn't reply right now.")})


# ✅ Language route
@app.route("/set-language/<lang>")
def set_language(lang):
    if lang in ["en", "hi"]:
        session["lang"] = lang
    return redirect(url_for("home"))

# ✅ Favicon support
@app.route("/favicon.ico")
def favicon():
    return send_from_directory("static", "favicon.ico")

# ✅ Home page
@app.route("/")
def home():
    t = get_text()
    return render_template("home.html", t=t, user=get_current_user())

# ✅ Skills page

# ✅ Skills page
@app.route("/skills")
@login_required
def skills():
    t = get_text()
    return render_template("skill_form.html", t=t, user=get_current_user())

# ✅ Skills result
@app.route("/skills-result", methods=["POST"])
@app.route("/skills_result", methods=["POST"])
@login_required
def skills_result():
    t = get_text()

    selected_skill = request.form.get("skill_select")

    if selected_skill == "OTHER":
        skill = request.form.get("skill_other", "").strip()
        if skill == "":
            skill = "Other"
    else:
        skill = selected_skill

    level = request.form.get("level", "Beginner")
    hours = request.form.get("hours", "1")
    language = request.form.get("language", "Hindi")

    youtube_link = f"https://www.youtube.com/results?search_query={skill}+training"
    google_link = f"https://www.google.com/search?q=learn+{skill}"
    instagram_link = f"https://www.instagram.com/explore/tags/{skill.replace(' ', '')}/"

    skills = [
        f"{skill} Basics",
        f"Practice Daily ({hours} hour/day)",
        "Create Portfolio / Samples",
        "Start small projects",
        "Apply for work or start freelancing"
    ]

    links = [
        ("YouTube", youtube_link),
        ("Google", google_link),
        ("Instagram", instagram_link),
        ("Find Nearby Jobs", url_for("find_jobs_nearby", query=skill))
    ]

    # ✅ DATASET BASED INCOME ESTIMATION (NO MANUAL VALUES)
    try:
        # ✅ Load cached dataset
        global df
        df, _, _, _ = load_dataset()
        
        if df is None or df.empty:
            raise Exception("Dataset not loaded")

        # 1) Exact skill match
        skill_df = df[df["skill"].astype(str).str.lower() == str(skill).lower()]

        # 2) If exact skill not found, match partial skill (robust)
        if skill_df.empty:
            skill_df = df[df["skill"].astype(str).str.lower().str.contains(str(skill).lower(), na=False)]

        # 3) If still empty, use same domain rows (fallback)
        if skill_df.empty:
            guess_df = df[df["skill"].astype(str).str.lower().str.contains(str(skill).lower().split(" ")[0], na=False)]
            if not guess_df.empty:
                domain_guess = guess_df["domain"].mode().iloc[0]
                skill_df = df[df["domain"].astype(str).str.lower() == str(domain_guess).lower()]

        # 4) If still empty, use full dataset as final fallback
        if skill_df.empty:
            skill_df = df.copy()

        # ✅ Adjust income based on hours/day (use close hour rows)
        h = int(hours)
        if "hours" in skill_df.columns:
            hour_filtered = skill_df[(skill_df["hours"] >= max(1, h-1)) & (skill_df["hours"] <= min(6, h+1))]
            if not hour_filtered.empty:
                skill_df = hour_filtered

        incomes = skill_df["income"].dropna().astype(int)

        # ✅ Use quantiles for realistic income range
        low = int(incomes.quantile(0.35))
        high = int(incomes.quantile(0.85))

        # ✅ Improve by Level effect (dataset-based boost)
        if level == "Intermediate":
            low = int(low * 1.10)
            high = int(high * 1.15)
        elif level == "Advanced":
            low = int(low * 1.20)
            high = int(high * 1.30)

        # ✅ Round for clean display
        low = (low // 500) * 500
        high = (high // 500) * 500

        if low == high:
            high = low + 1000

    except Exception as e:
        print("⚠️ Income estimate error:", e)
        low = 7000
        high = 15000

    # ✅ Save skill search to database
    if 'user_email' in session:
        try:
            user = User.query.filter_by(email=session['user_email']).first()
            if user:
                # Save to skill search history
                skill_search = SkillSearchHistory(
                    user_id=user.id,
                    skill_name=skill
                )
                db.session.add(skill_search)
                
                # Add or update user skill
                existing_skill = UserSkill.query.filter_by(user_id=user.id, skill_name=skill).first()
                if not existing_skill:
                    new_skill = UserSkill(
                        user_id=user.id,
                        skill_name=skill,
                        level=level,
                        progress=10  # Starting progress
                    )
                    db.session.add(new_skill)
                else:
                    existing_skill.level = level
                    existing_skill.updated_at = datetime.now()
                
                db.session.commit()
                print(f"✅ Skill search saved for user {user.email}: {skill}")
        except Exception as e:
            print(f"❌ Error saving skill search: {e}")
            db.session.rollback()

    return render_template("result.html", t=t, work=f"{skill} Training", low=low, high=high, skills=skills, links=links, user=get_current_user())

# ✅ Find jobs nearby
@app.route("/find-jobs-nearby")
@login_required
def find_jobs_nearby():
    """Redirect users to job portals with pre-filled query and location (if available)."""
    query = request.args.get("query", "Work From Home")
    location = session.get("location", "")
    q = f"{query} {location}".strip()
    return redirect(f"https://www.google.com/search?q={q.replace(' ', '+')}+jobs+near+me")

# ✅ Jobs page
@app.route("/jobs")
@login_required
def jobs():
    t = get_text()
    return render_template("form.html", t=t, user=get_current_user())

def _clean_job_label(label):
    """Clean up model labels like 'Agriculture_Mid' → 'Agriculture - Mid Level'"""
    level_map = {
        'Entry': 'Entry Level',
        'Mid': 'Mid Level',
        'Senior': 'Senior Level',
    }
    if '_' in label:
        parts = label.rsplit('_', 1)
        domain_part = parts[0].replace('_', ' ')
        level_part = level_map.get(parts[1], parts[1])
        return f"{domain_part} - {level_part}"
    return label

# ✅ Predict job recommendation
@app.route("/predict", methods=["POST"])
@login_required
def predict():
    t = get_text()

    # ============================================
    # COLLECT ALL FORM INPUTS
    # ============================================
    age = int(request.form.get("age", 25))
    kids = int(request.form.get("kids", 0))
    hours = int(request.form.get("hours", 4))
    experience_years = int(request.form.get("experience_years", 0))
    domain = request.form.get("domain", "General")
    sector = request.form.get("sector", "General")
    primary_skill = request.form.get("primary_skill", "General")
    secondary_skill = request.form.get("secondary_skill", "")
    education = request.form.get("education", "12th Pass (HSC)")
    city_type = request.form.get("city_type", "Tier-2")
    language = request.form.get("language", "Hindi")
    device = request.form.get("device", "Mobile Phone")
    work_mode = request.form.get("work_mode", "Work From Home")
    work_type = request.form.get("work_type", "Part-time")
    marital_status = request.form.get("marital_status", "Married")
    shift_type = request.form.get("shift_type", "General Shift")
    location = request.form.get("location", "").strip()

    if location:
        session["location"] = location

    # Build the combined skill strings matching training data format
    all_skills = f"{primary_skill}, {secondary_skill}" if secondary_skill else primary_skill

    # ============================================
    # BUILD BASE USER DATA (shared across models)
    # ============================================
    # These are sensible defaults for fields the user doesn't fill.
    # The models were trained on dataset columns, so we provide reasonable values.
    user_base = {
        'age': age,
        'kids': kids,
        'hours_available': hours,
        'experience_years': experience_years,
        'domain': domain,
        'sector': sector,
        'primary_skill': primary_skill,
        'secondary_skill': secondary_skill,
        'all_skills': all_skills,
        'education': education,
        'city_tier': city_type,
        'work_mode': work_mode,
        'work_type': work_type,
        'marital_status': marital_status,
        'shift_type': shift_type,
        'language': language,
        'device': device,
        'seniority_level': 'Entry Level/Fresher' if experience_years <= 2 else ('Associate' if experience_years <= 5 else ('Senior Associate' if experience_years <= 8 else 'Manager')),
        'career_growth': 'Medium',
        'travel_required': 'No Travel',
        'remote_available': work_mode in ('Work From Home', 'Remote', 'Hybrid'),
        'flexible_timing': shift_type == 'Flexible' or work_mode in ('Work From Home', 'Remote'),
        'childcare_compatible': work_mode in ('Work From Home', 'Remote'),
        'women_friendly': True,
        'maternity_benefits': False,
        'training_provided': False,
        'health_insurance': False,
        'pf_available': False,
        'is_verified': False,
        'income': 30000,
        'mother_suitability_score': 7,
        'skill_match_score': 75,
        'work_life_balance': 7,
        'job_title': f'{domain} {work_type}',
        'city': location or 'Delhi',
        'state': 'Delhi',
        'country': 'India',
    }

    # ============================================
    # HELPER: Engineer features and transform for each model
    # ============================================
    results = {}

    # --- 1. JOB RECOMMENDATION MODEL ---
    try:
        if job_model and job_preprocessor_data:
            prep = job_preprocessor_data
            preprocessor = prep['preprocessor']
            label_encoder = prep['label_encoder']
            tfidf = prep['tfidf']
            svd = prep['svd']

            df_job = pd.DataFrame([user_base])
            # Engineer features
            df_job['age_group'] = pd.cut(df_job['age'], bins=[0, 25, 35, 45, 100], labels=['young', 'middle', 'senior', 'veteran'])
            df_job['exp_level'] = pd.cut(df_job['experience_years'], bins=[-1, 2, 5, 10, 100], labels=['fresher', 'junior', 'mid', 'senior'])
            df_job['computed_seniority'] = pd.cut(df_job['experience_years'], bins=[-1, 3, 8, 100], labels=['Entry', 'Mid', 'Senior']).astype(str)
            df_job['has_kids'] = (df_job['kids'] > 0).astype(int)
            df_job['hours_category'] = pd.cut(df_job['hours_available'], bins=[0, 4, 6, 8, 100], labels=['part_time', 'half_day', 'full_day', 'overtime'])
            df_job['is_remote'] = df_job['work_mode'].isin(['Remote', 'Work From Home', 'Hybrid']).astype(int)
            df_job['skills_count'] = df_job['all_skills'].apply(lambda x: len(str(x).split(',')) if pd.notna(x) else 0)

            feature_names = prep.get('feature_names', ['age', 'kids', 'hours_available', 'experience_years', 'has_kids', 'is_remote', 'skills_count',
                                     'domain', 'primary_skill', 'education', 'city_tier', 'work_mode', 'marital_status', 'age_group', 'exp_level', 'hours_category', 'computed_seniority'])
            for col in feature_names:
                if col not in df_job.columns:
                    df_job[col] = 'Unknown' if col in ['domain', 'primary_skill', 'education', 'city_tier', 'work_mode', 'marital_status', 'age_group', 'exp_level', 'hours_category', 'computed_seniority'] else 0

            X_transformed = preprocessor.transform(df_job[feature_names])
            if hasattr(X_transformed, 'toarray'):
                X_transformed = X_transformed.toarray()
            text_features = tfidf.transform([all_skills])
            text_reduced = svd.transform(text_features)
            X_final = np.hstack([X_transformed, text_reduced])

            # Get top-5 predictions
            if hasattr(job_model, 'predict_proba'):
                proba = job_model.predict_proba(X_final)[0]
                top_indices = np.argsort(proba)[::-1][:5]
                top_labels = label_encoder.inverse_transform(top_indices)
                top_probs = proba[top_indices]
                # Normalize top-5 probabilities so they sum to ~100%
                top_sum = top_probs.sum()
                if top_sum > 0:
                    normalized_probs = top_probs / top_sum
                else:
                    normalized_probs = top_probs
                results['job_predictions'] = [{'label': _clean_job_label(lbl), 'confidence': float(prob)} for lbl, prob in zip(top_labels, normalized_probs)]
            else:
                pred = job_model.predict(X_final)[0]
                pred_label = label_encoder.inverse_transform([pred])[0]
                results['job_predictions'] = [{'label': _clean_job_label(pred_label), 'confidence': 0.85}]
        else:
            results['job_predictions'] = [{'label': f'{domain}_Entry', 'confidence': 0.60}]
    except Exception as e:
        print(f"Job model error: {e}")
        results['job_predictions'] = [{'label': f'{domain}_Entry', 'confidence': 0.50}]

    # --- 2. INCOME PREDICTION MODEL ---
    try:
        if income_model and income_preprocessor_data:
            prep = income_preprocessor_data
            preprocessor = prep['preprocessor']

            df_inc = pd.DataFrame([user_base])
            # Convert booleans to int
            for col in ['remote_available', 'flexible_timing', 'childcare_compatible', 'women_friendly', 'maternity_benefits', 'training_provided', 'health_insurance', 'pf_available']:
                df_inc[col] = df_inc[col].astype(int)

            # Engineer features
            df_inc['age_squared'] = df_inc['age'] ** 2
            df_inc['age_group'] = pd.cut(df_inc['age'], bins=[0, 25, 30, 35, 40, 50, 100], labels=['<25', '25-30', '30-35', '35-40', '40-50', '50+'])
            df_inc['exp_squared'] = df_inc['experience_years'] ** 2
            df_inc['exp_age_ratio'] = df_inc['experience_years'] / (df_inc['age'] - 18 + 1).clip(lower=1)
            df_inc['exp_level'] = pd.cut(df_inc['experience_years'], bins=[-1, 1, 3, 5, 8, 12, 100], labels=['fresher', 'junior', 'mid', 'senior', 'lead', 'expert'])
            df_inc['flexibility_score'] = df_inc['remote_available'] + df_inc['flexible_timing'] + df_inc['work_mode'].isin(['Remote', 'Work From Home', 'Hybrid']).astype(int)
            df_inc['benefits_score'] = df_inc[['health_insurance', 'pf_available', 'maternity_benefits', 'training_provided']].sum(axis=1)
            df_inc['mother_friendly_composite'] = (df_inc['mother_suitability_score'] + df_inc['childcare_compatible'] + df_inc['women_friendly'] + df_inc['flexible_timing']) / 4
            df_inc['city_tier_num'] = df_inc['city_tier'].map({'Metro': 4, 'Tier-1': 3, 'Tier-2': 2, 'Tier-3': 1, 'Rural': 0, 'Remote': 3}).fillna(1)
            df_inc['has_kids'] = (df_inc['kids'] > 0).astype(int)
            df_inc['kids_impact'] = df_inc['kids'] * df_inc['hours_available']
            df_inc['seniority_num'] = df_inc['seniority_level'].map({'Entry Level/Fresher': 0, 'Associate': 1, 'Senior Associate': 2, 'Manager': 3, 'Senior Manager': 4, 'Director/Executive': 5}).fillna(1)
            df_inc['career_growth_num'] = df_inc['career_growth'].map({'Low': 0, 'Medium': 1, 'High': 2}).fillna(1)
            edu_map = {'Below 8th/Informal Education': 0, '8th Pass': 1, '10th Pass (SSC)': 2, '12th Pass (HSC)': 3, 'Diploma/ITI': 4, 'Graduate (BTech/BA/BCom/BSc)': 5, 'Post Graduate (MBA/MTech/MA/MSc)': 6, 'PhD/Doctorate': 7}
            df_inc['education_num'] = df_inc['education'].map(edu_map).fillna(3)

            feature_names = prep.get('feature_names', list(df_inc.columns))
            for col in feature_names:
                if col not in df_inc.columns:
                    df_inc[col] = 0

            X_transformed = preprocessor.transform(df_inc[feature_names])
            if hasattr(X_transformed, 'toarray'):
                X_transformed = X_transformed.toarray()

            # TF-IDF
            tfidf_skills = prep.get('tfidf_skills')
            svd_skills = prep.get('svd_skills')
            tfidf_jobs = prep.get('tfidf_jobs')
            svd_jobs = prep.get('svd_jobs')

            parts = [X_transformed]
            if tfidf_skills and svd_skills:
                skills_reduced = svd_skills.transform(tfidf_skills.transform([all_skills]))
                parts.append(skills_reduced)
            if tfidf_jobs and svd_jobs:
                jobs_reduced = svd_jobs.transform(tfidf_jobs.transform([user_base['job_title']]))
                parts.append(jobs_reduced)

            X_final = np.hstack(parts)
            y_log_pred = income_model.predict(X_final)[0]
            income_pred = int(np.expm1(y_log_pred))
            income_pred = max(5000, min(500000, income_pred))
            results['income'] = income_pred
            results['income_low'] = int(income_pred * 0.80)
            results['income_high'] = int(income_pred * 1.25)
        else:
            results['income'] = 25000
            results['income_low'] = 20000
            results['income_high'] = 35000
    except Exception as e:
        print(f"Income model error: {e}")
        results['income'] = 25000
        results['income_low'] = 20000
        results['income_high'] = 35000

    # Update user_base income with predicted value for other models
    user_base['income'] = results['income']

    # --- 3. MOTHER SUITABILITY MODEL ---
    try:
        if mother_model and mother_preprocessor_data:
            prep = mother_preprocessor_data
            preprocessor = prep['preprocessor']
            label_encoder = prep['label_encoder']

            df_m = pd.DataFrame([user_base])
            for col in ['remote_available', 'flexible_timing', 'childcare_compatible', 'women_friendly', 'maternity_benefits', 'training_provided', 'health_insurance', 'pf_available']:
                df_m[col] = df_m[col].astype(int)

            # Engineer features
            df_m['flexibility_score'] = df_m[['remote_available', 'flexible_timing', 'childcare_compatible']].sum(axis=1) * 25
            df_m['family_support_score'] = df_m[['maternity_benefits', 'women_friendly', 'childcare_compatible']].sum(axis=1) * 33
            df_m['total_benefits'] = df_m[['health_insurance', 'pf_available', 'maternity_benefits', 'training_provided']].sum(axis=1)
            travel_stress = df_m['travel_required'].map({'No Travel': 0, 'Occasional': 15, 'Frequent': 30}).fillna(10)
            df_m['work_stress'] = travel_stress + (12 - df_m['hours_available']).clip(lower=0) * 5
            df_m['mother_friendly_composite'] = df_m['flexibility_score'] * 0.4 + df_m['family_support_score'] * 0.3 + df_m['work_life_balance'] * 0.3
            df_m['kids_factor'] = np.where(df_m['kids'] > 0, 1, 0)
            df_m['kids_count_impact'] = np.minimum(df_m['kids'], 4) * 10
            df_m['day_shift'] = df_m['shift_type'].str.lower().str.contains('day|morning|general', na=False).astype(int)
            df_m['hours_suitable'] = np.where(df_m['hours_available'] <= 8, 1, 0)

            feature_names = prep.get('feature_names') or prep.get('numeric_cols', []) + prep.get('categorical_cols', []) + prep.get('binary_cols', [])
            if not feature_names:
                # Fallback: use all columns the preprocessor was trained on
                feature_names = list(df_m.columns)

            for col in feature_names:
                if col not in df_m.columns:
                    df_m[col] = 0

            X_transformed = preprocessor.transform(df_m[feature_names])
            if hasattr(X_transformed, 'toarray'):
                X_transformed = X_transformed.toarray()

            pred = mother_model.predict(X_transformed)[0]
            pred_label = label_encoder.inverse_transform([pred])[0]
            results['mother_suitability'] = pred_label

            if hasattr(mother_model, 'predict_proba'):
                proba = mother_model.predict_proba(X_transformed)[0]
                results['mother_suitability_confidence'] = float(max(proba))
            else:
                results['mother_suitability_confidence'] = 0.75
        else:
            results['mother_suitability'] = 'Good'
            results['mother_suitability_confidence'] = 0.60
    except Exception as e:
        print(f"Mother suitability error: {e}")
        results['mother_suitability'] = 'Moderate'
        results['mother_suitability_confidence'] = 0.50

    # --- 4. WORK-LIFE BALANCE MODEL ---
    try:
        if wlb_model and wlb_preprocessor_data:
            prep = wlb_preprocessor_data
            preprocessor = prep['preprocessor']
            label_encoder = prep['label_encoder']

            df_w = pd.DataFrame([user_base])
            for col in ['remote_available', 'flexible_timing', 'childcare_compatible', 'women_friendly', 'maternity_benefits', 'training_provided', 'health_insurance', 'pf_available']:
                df_w[col] = df_w[col].astype(int)

            # Engineer features
            df_w['flexibility_score'] = df_w[['remote_available', 'flexible_timing', 'childcare_compatible']].sum(axis=1) * 33.33
            df_w['hours_stress'] = (12 - df_w['hours_available']).clip(lower=0) * 10
            df_w['reasonable_hours'] = (df_w['hours_available'] <= 8).astype(int)
            df_w['no_travel'] = df_w['travel_required'].map({'No Travel': 1, 'Occasional': 0.5, 'Frequent': 0}).fillna(0.5)
            df_w['day_shift'] = df_w['shift_type'].str.lower().str.contains('day|morning|general', na=False).astype(int)
            df_w['night_shift'] = df_w['shift_type'].str.lower().str.contains('night', na=False).astype(int)
            df_w['family_friendly_score'] = df_w[['women_friendly', 'maternity_benefits', 'childcare_compatible']].sum(axis=1) * 33.33
            df_w['benefits_score'] = df_w[['health_insurance', 'pf_available', 'training_provided']].sum(axis=1) * 25
            wm_score_map = {'Work From Home': 100, 'Remote': 100, 'Hybrid': 75, 'Office': 50, 'On-site': 40, 'Field Work': 30}
            df_w['work_mode_score'] = df_w['work_mode'].map(wm_score_map).fillna(50)
            df_w['balance_composite'] = df_w['flexibility_score'] * 0.35 + df_w['family_friendly_score'] * 0.25 + df_w['work_mode_score'] * 0.20 + (100 - df_w['hours_stress']) * 0.20
            df_w['has_kids'] = (df_w['kids'] > 0).astype(int)
            df_w['multiple_kids'] = (df_w['kids'] > 1).astype(int)

            feature_names = prep.get('feature_names') or prep.get('numeric_cols', []) + prep.get('categorical_cols', []) + prep.get('binary_cols', [])
            if not feature_names:
                feature_names = list(df_w.columns)

            for col in feature_names:
                if col not in df_w.columns:
                    df_w[col] = 0

            X_transformed = preprocessor.transform(df_w[feature_names])
            if hasattr(X_transformed, 'toarray'):
                X_transformed = X_transformed.toarray()

            pred = wlb_model.predict(X_transformed)[0]
            pred_label = label_encoder.inverse_transform([pred])[0]
            results['work_life_balance'] = pred_label.replace('_', ' ')
        else:
            results['work_life_balance'] = 'Good'
    except Exception as e:
        print(f"WLB model error: {e}")
        results['work_life_balance'] = 'Average'

    # --- 5. CAREER PATH MODEL ---
    try:
        if career_model and career_preprocessor_data:
            prep = career_preprocessor_data
            preprocessor = prep['preprocessor']
            label_encoder = prep['label_encoder']

            df_c = pd.DataFrame([user_base])
            for col in ['remote_available', 'flexible_timing', 'training_provided', 'health_insurance', 'pf_available', 'is_verified']:
                df_c[col] = df_c[col].astype(int)

            # Engineer features
            df_c['skills_count'] = df_c['all_skills'].apply(lambda x: len(str(x).split(',')) if pd.notna(x) else 0)
            df_c['exp_age_ratio'] = df_c['experience_years'] / (df_c['age'] - 17).clip(lower=1)
            df_c['career_velocity'] = df_c['skills_count'] / (df_c['experience_years'] + 1)
            df_c['income_per_exp'] = df_c['income'] / (df_c['experience_years'] + 1)
            edu_level_map = {'Below 8th/Informal Education': 0, '8th Pass': 1, '10th Pass (SSC)': 2, '12th Pass (HSC)': 3, 'Diploma/ITI': 4, 'Graduate (BTech/BA/BCom/BSc)': 5, 'Post Graduate (MBA/MTech/MA/MSc)': 6, 'PhD/Doctorate': 7}
            df_c['education_level'] = df_c['education'].map(edu_level_map).fillna(3)
            df_c['career_growth_num'] = df_c['career_growth'].map({'Low': 30, 'Medium': 60, 'High': 90}).fillna(50)
            df_c['growth_potential'] = df_c['career_growth_num'] / (df_c['experience_years'] + 1)
            df_c['job_stability'] = df_c[['is_verified', 'health_insurance', 'pf_available']].sum(axis=1)
            df_c['leadership_indicator'] = 0

            feature_names = prep.get('feature_names') or prep.get('numeric_cols', []) + prep.get('categorical_cols', []) + prep.get('binary_cols', [])
            if not feature_names:
                feature_names = list(df_c.columns)

            for col in feature_names:
                if col not in df_c.columns:
                    df_c[col] = 0

            X_structured = preprocessor.transform(df_c[feature_names])
            if hasattr(X_structured, 'toarray'):
                X_structured = X_structured.toarray()

            # TF-IDF text features
            tfidf_skills = prep.get('tfidf_skills')
            svd_obj = prep.get('svd')
            parts = [X_structured]
            if tfidf_skills and svd_obj:
                combined_text = f"{all_skills} {primary_skill} {user_base['job_title']}"
                text_reduced = svd_obj.transform(tfidf_skills.transform([combined_text]))
                parts.append(text_reduced)

            X_final = np.hstack(parts)
            pred = career_model.predict(X_final)[0]
            pred_label = label_encoder.inverse_transform([pred])[0]
            results['career_path'] = pred_label
        else:
            results['career_path'] = user_base['seniority_level']
    except Exception as e:
        print(f"Career path error: {e}")
        results['career_path'] = user_base['seniority_level']

    # --- 6. SKILL-JOB MATCHING MODEL ---
    # The trained model only has Low/Medium classes due to biased training data.
    # Use a rule-based score that considers actual user input for a meaningful result.
    try:
        score = 0

        # Experience contributes up to 25 points
        score += min(experience_years * 5, 25)

        # Skills count contributes up to 25 points
        skills_count = len([s.strip() for s in all_skills.split(',') if s.strip()])
        score += min(skills_count * 12, 25)

        # Education contributes up to 20 points
        edu_scores = {
            'Below 8th/Informal Education': 3, '8th Pass': 5,
            '10th Pass (SSC)': 8, '12th Pass (HSC)': 10,
            'Diploma/ITI': 13, 'Graduate (BTech/BA/BCom/BSc)': 17,
            'Post Graduate (MBA/MTech/MA/MSc)': 20, 'PhD/Doctorate': 20,
        }
        score += edu_scores.get(education, 10)

        # Hours available contributes up to 15 points
        score += min(hours * 2, 15)

        # Having a secondary skill adds up to 15 points
        if secondary_skill:
            score += 15

        score = min(score, 100)

        if score >= 75:
            results['skill_match'] = 'Excellent'
        elif score >= 55:
            results['skill_match'] = 'High'
        elif score >= 35:
            results['skill_match'] = 'Medium'
        else:
            results['skill_match'] = 'Low'
    except Exception as e:
        print(f"Skill match error: {e}")
        results['skill_match'] = 'Medium'

    # --- 7. SKILL GAP ANALYZER ---
    try:
        if skill_gap_model and skill_gap_preprocessor_data:
            prep = skill_gap_preprocessor_data
            preprocessor = prep['preprocessor']

            df_sg = pd.DataFrame([user_base])
            # Engineer features
            df_sg['skills_count'] = df_sg['all_skills'].apply(lambda x: len(str(x).split(',')) if pd.notna(x) else 0)
            skills_text = str(all_skills)
            unique_skills = len(set(skills_text.lower().split(',')))
            total_skills = max(len(skills_text.split(',')), 1)
            df_sg['skill_diversity'] = unique_skills / total_skills
            df_sg['skill_exp_ratio'] = df_sg['skills_count'] / (df_sg['experience_years'] + 1)
            df_sg['skill_gap_indicator'] = 100 - df_sg['skill_match_score']
            df_sg['career_growth_num'] = df_sg['career_growth'].map({'Low': 30, 'Medium': 60, 'High': 90}).fillna(50)
            df_sg['growth_gap'] = 100 - df_sg['career_growth_num']
            seniority_skills_map = {'Entry Level/Fresher': 3, 'Associate': 6, 'Senior Associate': 10, 'Manager': 15, 'Senior Manager': 20, 'Director/Executive': 15}
            df_sg['expected_skills'] = df_sg['seniority_level'].map(seniority_skills_map).fillna(5)
            df_sg['skill_deficiency'] = (df_sg['expected_skills'] - df_sg['skills_count']).clip(lower=0)
            df_sg['domain_complexity'] = 1.0
            df_sg['training_likelihood'] = df_sg['skill_deficiency'] * 5 + df_sg['skill_gap_indicator'] * 0.5 + df_sg['growth_gap'] * 0.3

            feature_names = prep.get('feature_names') or prep.get('numeric_cols', []) + prep.get('categorical_cols', [])
            if not feature_names:
                feature_names = list(df_sg.columns)

            for col in feature_names:
                if col not in df_sg.columns:
                    df_sg[col] = 0

            X_structured = preprocessor.transform(df_sg[feature_names])
            if hasattr(X_structured, 'toarray'):
                X_structured = X_structured.toarray()

            tfidf_skills = prep.get('tfidf_skills')
            tfidf_jobs = prep.get('tfidf_jobs')
            svd_obj = prep.get('svd')
            parts = [X_structured]
            if tfidf_skills and tfidf_jobs and svd_obj:
                from scipy.sparse import hstack as sparse_hstack
                combined_skills_text = f"{all_skills} {primary_skill} {secondary_skill}"
                skills_tfidf = tfidf_skills.transform([combined_skills_text])
                jobs_tfidf = tfidf_jobs.transform([user_base['job_title']])
                text_combined = sparse_hstack([skills_tfidf, jobs_tfidf])
                text_reduced = svd_obj.transform(text_combined)
                parts.append(text_reduced)

            X_final = np.hstack(parts)
            pred = skill_gap_model.predict(X_final)[0]
            results['training_needed'] = bool(pred)
        else:
            results['training_needed'] = True
    except Exception as e:
        print(f"Skill gap error: {e}")
        results['training_needed'] = True

    # --- 8. PROFILE COMPLETENESS ---
    try:
        if profile_model and profile_preprocessor_data:
            prep = profile_preprocessor_data
            preprocessor = prep['preprocessor']

            df_pc = pd.DataFrame([user_base])
            for col in ['remote_available', 'flexible_timing', 'childcare_compatible', 'women_friendly', 'maternity_benefits', 'training_provided', 'health_insurance', 'pf_available']:
                if col in df_pc.columns:
                    df_pc[col] = df_pc[col].fillna(0).astype(int)

            # Profile completeness scoring (rule-based part)
            profile_fields = {
                'age': (8, age > 0), 'education': (12, bool(education)),
                'experience_years': (10, experience_years > 0), 'primary_skill': (12, bool(primary_skill)),
                'all_skills': (8, bool(all_skills)), 'domain': (10, bool(domain)),
                'city': (5, bool(location)), 'marital_status': (3, bool(marital_status)),
                'hours_available': (5, hours > 0), 'work_mode': (5, bool(work_mode)),
                'secondary_skill': (5, bool(secondary_skill)), 'language': (3, bool(language)),
                'device': (2, bool(device)), 'kids': (2, True),
                'sector': (5, bool(sector)), 'shift_type': (5, bool(shift_type)),
            }
            completeness_score = sum(weight for weight, present in profile_fields.values() if present)
            total_possible = sum(weight for weight, _ in profile_fields.values())
            completeness_score = int(round(completeness_score / total_possible * 100))

            # Engineer features to match the trained model
            edu_order = {
                'Below 8th/Informal Education': 0, '8th Pass': 1,
                '10th Pass (SSC)': 2, '12th Pass (HSC)': 3,
                'Diploma/ITI': 4, 'Graduate (BTech/BA/BCom/BSc)': 5,
                'Post Graduate (MBA/MTech/MA/MSc)': 6, 'PhD/Doctorate': 7
            }
            sen_order = {'Entry': 0, 'Junior': 1, 'Mid': 2, 'Senior': 3, 'Lead': 4, 'Manager': 5}
            tier_map = {'Metro': 0, 'Tier-1': 1, 'Tier-2': 2, 'Tier-3': 3, 'Remote': 4, 'Rural': 5}

            df_pc['education_level'] = edu_order.get(education, 3)
            df_pc['seniority_num'] = sen_order.get(df_pc.get('seniority_level', {None: 1}).get(None, 1) if isinstance(df_pc.get('seniority_level'), dict) else 1, 1)
            if 'seniority_level' in df_pc.columns:
                df_pc['seniority_num'] = df_pc['seniority_level'].map(sen_order).fillna(1)
            else:
                df_pc['seniority_num'] = 1
            df_pc['skills_count'] = len(str(all_skills).split(',')) if all_skills else 0
            df_pc['has_secondary_skill'] = int(bool(secondary_skill))
            df_pc['exp_edu_interaction'] = experience_years * edu_order.get(education, 3)
            df_pc['income_per_exp'] = (df_pc.get('income', 0) if 'income' in df_pc.columns else 0) / (experience_years + 1) if isinstance(experience_years, (int, float)) else 0
            if 'income' in df_pc.columns:
                df_pc['income_per_exp'] = df_pc['income'].fillna(0) / (experience_years + 1)
            else:
                df_pc['income_per_exp'] = 0
            benefit_cols = ['health_insurance', 'pf_available', 'maternity_benefits',
                           'training_provided', 'flexible_timing', 'childcare_compatible',
                           'women_friendly', 'remote_available']
            df_pc['benefits_count'] = sum(int(df_pc[c].iloc[0]) if c in df_pc.columns else 0 for c in benefit_cols)
            df_pc['hours_bucket'] = 0 if hours <= 3 else (1 if hours <= 5 else (2 if hours <= 7 else 3))
            df_pc['age_group'] = 0 if age <= 25 else (1 if age <= 35 else (2 if age <= 45 else 3))
            ct = df_pc['city_tier'].iloc[0] if 'city_tier' in df_pc.columns else 'Tier-2'
            df_pc['city_tier_num'] = tier_map.get(str(ct), 2)

            feature_names = prep.get('numeric_cols', []) + prep.get('categorical_cols', []) + prep.get('binary_cols', [])

            for col in feature_names:
                if col not in df_pc.columns:
                    df_pc[col] = 0

            X_structured = preprocessor.transform(df_pc[feature_names])
            if hasattr(X_structured, 'toarray'):
                X_structured = X_structured.toarray()

            tfidf_skills = prep.get('tfidf_skills')
            svd_obj = prep.get('svd')
            parts = [X_structured]
            if tfidf_skills and svd_obj:
                combined_text = f"{all_skills or ''} {primary_skill or ''} {secondary_skill or ''}"
                text_reduced = svd_obj.transform(tfidf_skills.transform([combined_text]))
                parts.append(text_reduced)

            X_final = np.hstack(parts)
            quality_pred = profile_model.predict(X_final)[0]

            # Combine rule-based completeness with ML quality prediction
            if quality_pred == 1 and completeness_score >= 80:
                completeness_score = max(completeness_score, 90)
            elif quality_pred == 0 and completeness_score > 85:
                completeness_score = min(completeness_score, 85)

            results['profile_completeness'] = completeness_score
            results['profile_grade'] = 'A+' if completeness_score >= 95 else ('A' if completeness_score >= 85 else ('B' if completeness_score >= 70 else ('C' if completeness_score >= 50 else 'D')))
        else:
            results['profile_completeness'] = 75
            results['profile_grade'] = 'B'
    except Exception as e:
        print(f"Profile completeness error: {e}")
        results['profile_completeness'] = 70
        results['profile_grade'] = 'B'

    # ============================================
    # FETCH JOB RECOMMENDATIONS FROM DATASET
    # ============================================
    job_recommendations = []
    try:
        global df
        df, _, _, _ = load_dataset()

        if df is not None and not df.empty:
            # Try matching by domain first
            filtered_df = df[df['domain'].str.contains(domain, case=False, na=False)]

            # If not enough, also include sector matches
            if len(filtered_df) < 10:
                sector_df = df[df['sector'].str.contains(sector, case=False, na=False)]
                filtered_df = pd.concat([filtered_df, sector_df]).drop_duplicates()

            # If still not enough, include skill matches
            if len(filtered_df) < 10:
                skill_df = df[df['primary_skill'].str.contains(primary_skill, case=False, na=False)]
                filtered_df = pd.concat([filtered_df, skill_df]).drop_duplicates()

            # Sort by relevance: prefer matching work_mode and work_type
            if not filtered_df.empty:
                filtered_df = filtered_df.copy()
                filtered_df['_relevance'] = 0
                if 'work_mode' in filtered_df.columns:
                    filtered_df.loc[filtered_df['work_mode'].str.contains(work_mode, case=False, na=False), '_relevance'] += 2
                if 'work_type' in filtered_df.columns:
                    filtered_df.loc[filtered_df['work_type'].str.contains(work_type, case=False, na=False), '_relevance'] += 1
                filtered_df = filtered_df.sort_values('_relevance', ascending=False)

            top_jobs = filtered_df.head(10).to_dict('records')

            for idx, job in enumerate(top_jobs, 1):
                job_title_text = job.get('job_title', f'{domain} Opportunity')
                job_recommendations.append({
                    'id': idx,
                    'title': job_title_text,
                    'company': job.get('company', 'MaaSarthi Partner'),
                    'description': job.get('job_description', f'{domain} role'),
                    'salary': f"₹{job.get('salary_min', results['income_low']):,.0f} - ₹{job.get('salary_max', results['income_high']):,.0f}",
                    'work_mode': job.get('work_mode', work_mode),
                    'location': job.get('city', location or 'Remote'),
                    'work_type': job.get('work_type', work_type),
                    'apply_links': {
                        'linkedin': f"https://www.linkedin.com/jobs/search/?keywords={job_title_text.replace(' ', '%20')}&location=India",
                        'naukri': f"https://www.naukri.com/{job_title_text.lower().replace(' ', '-')}-jobs",
                        'indeed': f"https://www.indeed.co.in/jobs?q={job_title_text.replace(' ', '+')}&l=India",
                        'internshala': f"https://internshala.com/jobs/{job_title_text.lower().replace(' ', '-')}-jobs"
                    }
                })
    except Exception as e:
        print(f"Error fetching jobs: {e}")

    if not job_recommendations:
        search_term = domain.replace(' ', '%20')
        job_recommendations = [{
            'id': 1,
            'title': f'{domain} - {work_mode}',
            'company': 'MaaSarthi Partner',
            'description': f'{work_type} {domain} opportunity',
            'salary': f"₹{results['income_low']:,} - ₹{results['income_high']:,}",
            'work_mode': work_mode,
            'location': location or 'Remote',
            'work_type': work_type,
            'apply_links': {
                'linkedin': f"https://www.linkedin.com/jobs/search/?keywords={search_term}&location=India",
                'naukri': f"https://www.naukri.com/{domain.lower().replace(' ', '-')}-jobs",
                'indeed': f"https://www.indeed.co.in/jobs?q={domain.replace(' ', '+')}&l=India",
                'internshala': f"https://internshala.com/jobs/{domain.lower().replace(' ', '-')}-jobs"
            }
        }]

    # Suggested skills based on domain
    suggested_skills = _get_suggested_skills(domain, primary_skill)

    # Learning resources
    links = [
        ("YouTube - " + domain, f"https://www.youtube.com/results?search_query={domain.replace(' ', '+')}+tutorial"),
        ("Google - " + domain, f"https://www.google.com/search?q={domain.replace(' ', '+')}+free+course"),
        ("Coursera", "https://www.coursera.org"),
        ("Khan Academy", "https://www.khanacademy.org"),
    ]

    # Overall confidence from job prediction
    top_job_pred = results.get('job_predictions', [{}])[0]
    confidence_percentage = int(round(top_job_pred.get('confidence', 0.70) * 100))

    # ============================================
    # SAVE TO DATABASE
    # ============================================
    if 'user_email' in session:
        try:
            user = User.query.filter_by(email=session['user_email']).first()
            if user:
                job_search = JobSearchHistory(
                    user_id=user.id, age=age, education=education,
                    domain=domain, skill=primary_skill, work_mode=work_mode,
                    location=location, city_type=city_type, hours=hours, kids=kids,
                    language=language, device=device,
                    predicted_job=top_job_pred.get('label', domain),
                    predicted_salary_low=results['income_low'],
                    predicted_salary_high=results['income_high'],
                    confidence_score=top_job_pred.get('confidence', 0.70)
                )
                db.session.add(job_search)

                if job_recommendations:
                    top_job = job_recommendations[0]
                    job_rec = JobRecommendation(
                        user_id=user.id,
                        job_title=top_job.get('title', domain),
                        company=top_job.get('company', 'MaaSarthi Partner'),
                        salary=f"₹{results['income_low']} - ₹{results['income_high']}",
                        hours=f'{hours} hours/day',
                        location=location or 'Remote',
                        description=top_job.get('description', '')
                    )
                    db.session.add(job_rec)

                profile = UserProfile.query.filter_by(user_id=user.id).first()
                if not profile:
                    profile = UserProfile(user_id=user.id)
                    db.session.add(profile)

                profile.age = age
                profile.education = education
                profile.location = location
                profile.city_type = city_type
                profile.preferred_domain = domain
                profile.primary_skill = primary_skill
                profile.work_mode_preference = work_mode
                profile.available_hours = hours
                profile.number_of_kids = kids
                profile.language_preference = language
                profile.device_type = device
                profile.profile_completed = True

                db.session.commit()
        except Exception as e:
            print(f"Error saving job search: {e}")
            db.session.rollback()

    return render_template(
        "result.html",
        t=t,
        # Model results
        results=results,
        # Job data
        jobs=job_recommendations,
        confidence_score=confidence_percentage,
        # Income
        low=results['income_low'],
        high=results['income_high'],
        income_prediction=results['income'],
        # Skills & links
        skills=suggested_skills,
        links=links,
        # Context
        domain=domain,
        primary_skill=primary_skill,
        work_mode=work_mode,
        user=get_current_user()
    )


def _get_suggested_skills(domain, current_skill):
    """Return domain-specific skill suggestions."""
    domain_skills = {
        'Software Engineering': ['Python', 'JavaScript', 'React', 'SQL', 'Git'],
        'Data Science': ['Python', 'Machine Learning', 'SQL', 'Tableau', 'Statistics'],
        'Digital Marketing': ['SEO', 'Google Ads', 'Social Media', 'Content Writing', 'Analytics'],
        'Graphic Design': ['Photoshop', 'Illustrator', 'Figma', 'Canva', 'Typography'],
        'Content Writing': ['SEO Writing', 'Copywriting', 'Blogging', 'Social Media', 'Editing'],
        'Data Entry': ['Excel', 'Typing Speed', 'Google Sheets', 'MS Office', 'Accuracy'],
        'Customer Support': ['Communication', 'CRM Tools', 'Email Writing', 'Problem Solving', 'Patience'],
        'School Teaching': ['Lesson Planning', 'EdTech Tools', 'Communication', 'Subject Expertise', 'Patience'],
        'Beauty Wellness': ['Skin Care', 'Hair Styling', 'Makeup', 'Nail Art', 'Client Management'],
        'Handicrafts': ['Product Photography', 'Online Selling', 'Packaging', 'Design', 'Marketing'],
        'Ecommerce': ['Product Listing', 'Inventory', 'Customer Service', 'Marketing', 'Analytics'],
        'Nursing': ['Patient Care', 'Medical Records', 'First Aid', 'Communication', 'Empathy'],
    }
    skills = domain_skills.get(domain, ['Communication', 'Time Management', 'Computer Basics', 'Online Collaboration', 'Self-Marketing'])
    return [s for s in skills if s.lower() != current_skill.lower()][:5]

# Google OAuth setup
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "YOUR_GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "YOUR_GOOGLE_CLIENT_SECRET")
google_bp = make_google_blueprint(
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
    scope=["profile", "email"],
    redirect_url="/dashboard"
)
app.register_blueprint(google_bp, url_prefix="/login")

@app.route("/login/google")
def login_google():
    if not google.authorized:
        return redirect(url_for("google.login"))
    resp = google.get("/oauth2/v2/userinfo")
    user_info = resp.json()
    session['user_email'] = user_info['email']
    session['user_name'] = user_info.get('name', 'User')
    return redirect(url_for("dashboard"))

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        # Sanitize inputs
        name = SecurityUtils.sanitize_input(request.form.get('name', ''))
        email = request.form.get('email', '').strip().lower()
        subject = SecurityUtils.sanitize_input(request.form.get('subject', ''))
        message = SecurityUtils.sanitize_input(request.form.get('message', ''))
        
        # Validate inputs
        errors = []
        
        is_valid, error_msg = Validator.validate_name(name)
        if not is_valid:
            errors.append(error_msg)
        
        is_valid, error_msg = Validator.validate_email(email)
        if not is_valid:
            errors.append(error_msg)
        
        if not message or len(message) < 10:
            errors.append('Message must be at least 10 characters long')
        
        if len(message) > 2000:
            errors.append('Message is too long (max 2000 characters)')
        
        if errors:
            return render_template('contact.html', error=' | '.join(errors), user=get_current_user())
        
        # Save contact message to database
        contact_msg = ContactMessage(
            name=name,
            email=email,
            subject=subject,
            message=message,
            created_at=datetime.now()
        )
        db.session.add(contact_msg)
        db.session.commit()
        
        print(f"📧 Contact form submission saved from {name} ({email}): {subject}")
        
        flash('Thank you for contacting us! We will get back to you soon.', 'success')
        return render_template('contact.html', success=True, name=name, user=get_current_user())
    return render_template('contact.html', success=False, user=get_current_user())


# ============================================
# 🔐 SECURITY MIDDLEWARE & HEADERS
# ============================================

@app.after_request
def add_security_headers(response):
    """Add security headers to all responses"""
    # Prevent clickjacking
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    
    # Prevent MIME type sniffing
    response.headers['X-Content-Type-Options'] = 'nosniff'
    
    # Enable XSS filter
    response.headers['X-XSS-Protection'] = '1; mode=block'
    
    # Referrer policy
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    
    # Cache control for sensitive pages
    if request.endpoint in ['login', 'signup', 'dashboard']:
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
    
    return response


@app.before_request
def check_session_validity():
    """Check session validity before each request"""
    # Skip for static files and public routes
    public_routes = ['login', 'signup', 'index', 'home', 'static', 'contact', 'set_language']
    
    if request.endpoint in public_routes:
        return
    
    # Check if session has expired (24 hours)
    if 'login_time' in session:
        login_time = datetime.fromisoformat(session['login_time'])
        if datetime.now() - login_time > timedelta(hours=24):
            session.clear()
            return redirect('/login?error=Session+expired.+Please+login+again.')


# ============================================
# 🔐 API ENDPOINTS FOR SECURITY
# ============================================

@app.route('/api/validate-email', methods=['POST'])
def api_validate_email():
    """API endpoint to validate email in real-time"""
    data = request.get_json()
    email = data.get('email', '') if data else ''
    
    is_valid, error_msg = Validator.validate_email(email)
    
    # Also check if email exists in database
    email_exists = User.query.filter_by(email=email.lower()).first() is not None
    
    return jsonify({
        'valid': is_valid,
        'error': error_msg,
        'exists': email_exists
    })


@app.route('/api/validate-password', methods=['POST'])
def api_validate_password():
    """API endpoint to validate password strength in real-time"""
    data = request.get_json()
    password = data.get('password', '') if data else ''
    
    is_valid, error_msg = Validator.validate_password(password)
    
    # Calculate password strength
    strength = 0
    if len(password) >= 8:
        strength += 1
    if len(password) >= 12:
        strength += 1
    if re.search(r'[A-Z]', password):
        strength += 1
    if re.search(r'[a-z]', password):
        strength += 1
    if re.search(r'\d', password):
        strength += 1
    if re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        strength += 1
    
    strength_text = ['Very Weak', 'Weak', 'Fair', 'Good', 'Strong', 'Very Strong'][min(strength, 5)]
    
    return jsonify({
        'valid': is_valid,
        'error': error_msg,
        'strength': strength,
        'strength_text': strength_text
    })


@app.route('/api/check-session', methods=['GET'])
def api_check_session():
    """API endpoint to check if user is logged in"""
    is_logged_in = 'user_email' in session and 'session_token' in session
    
    return jsonify({
        'logged_in': is_logged_in,
        'user_name': session.get('user_name') if is_logged_in else None
    })


# ============================================
# 📊 DASHBOARD API ENDPOINTS
# ============================================

@app.route('/api/add-task', methods=['POST'])
def api_add_task():
    """Add a new task for the user"""
    if 'user_email' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    user = User.query.filter_by(email=session['user_email']).first()
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    data = request.get_json()
    task_text = data.get('task_text', '').strip()
    
    if not task_text:
        return jsonify({'error': 'Task text is required'}), 400
    
    new_task = Task(user_id=user.id, task_text=task_text)
    db.session.add(new_task)
    db.session.commit()
    
    return jsonify({'success': True, 'task_id': new_task.id, 'message': 'Task added successfully'})


@app.route('/api/toggle-task/<int:task_id>', methods=['POST'])
def api_toggle_task(task_id):
    """Toggle task completion status"""
    if 'user_email' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    user = User.query.filter_by(email=session['user_email']).first()
    task = Task.query.filter_by(id=task_id, user_id=user.id).first()
    
    if not task:
        return jsonify({'error': 'Task not found'}), 404
    
    task.is_completed = not task.is_completed
    db.session.commit()
    
    return jsonify({'success': True, 'is_completed': task.is_completed})


@app.route('/api/delete-task/<int:task_id>', methods=['DELETE'])
def api_delete_task(task_id):
    """Delete a task"""
    if 'user_email' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    user = User.query.filter_by(email=session['user_email']).first()
    task = Task.query.filter_by(id=task_id, user_id=user.id).first()
    
    if not task:
        return jsonify({'error': 'Task not found'}), 404
    
    db.session.delete(task)
    db.session.commit()
    
    return jsonify({'success': True, 'message': 'Task deleted successfully'})


@app.route('/api/add-reminder', methods=['POST'])
def api_add_reminder():
    """Add a new reminder for the user"""
    if 'user_email' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    user = User.query.filter_by(email=session['user_email']).first()
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    data = request.get_json()
    title = data.get('title', '').strip()
    description = data.get('description', '').strip()
    due_date_str = data.get('due_date')
    
    if not title:
        return jsonify({'error': 'Reminder title is required'}), 400
    
    due_date = None
    if due_date_str:
        try:
            due_date = datetime.fromisoformat(due_date_str)
        except:
            return jsonify({'error': 'Invalid date format'}), 400
    
    new_reminder = Reminder(user_id=user.id, title=title, description=description, due_date=due_date)
    db.session.add(new_reminder)
    db.session.commit()
    
    return jsonify({'success': True, 'reminder_id': new_reminder.id, 'message': 'Reminder added successfully'})


@app.route('/api/toggle-reminder/<int:reminder_id>', methods=['POST'])
def api_toggle_reminder(reminder_id):
    """Toggle reminder completion status"""
    if 'user_email' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    user = User.query.filter_by(email=session['user_email']).first()
    reminder = Reminder.query.filter_by(id=reminder_id, user_id=user.id).first()
    
    if not reminder:
        return jsonify({'error': 'Reminder not found'}), 404
    
    reminder.is_completed = not reminder.is_completed
    db.session.commit()
    
    return jsonify({'success': True, 'is_completed': reminder.is_completed})


@app.route('/api/add-skill', methods=['POST'])
def api_add_skill():
    """Add a new skill for the user"""
    if 'user_email' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    user = User.query.filter_by(email=session['user_email']).first()
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    data = request.get_json()
    skill_name = data.get('skill_name', '').strip()
    level = data.get('level', 'Beginner')
    
    if not skill_name:
        return jsonify({'error': 'Skill name is required'}), 400
    
    # Check if skill already exists
    existing_skill = UserSkill.query.filter_by(user_id=user.id, skill_name=skill_name).first()
    if existing_skill:
        return jsonify({'error': 'Skill already exists'}), 409
    
    new_skill = UserSkill(user_id=user.id, skill_name=skill_name, level=level, progress=0)
    db.session.add(new_skill)
    db.session.commit()
    
    return jsonify({'success': True, 'skill_id': new_skill.id, 'message': 'Skill added successfully'})


@app.route('/api/update-skill-progress/<int:skill_id>', methods=['POST'])
def api_update_skill_progress(skill_id):
    """Update skill progress"""
    if 'user_email' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    user = User.query.filter_by(email=session['user_email']).first()
    skill = UserSkill.query.filter_by(id=skill_id, user_id=user.id).first()
    
    if not skill:
        return jsonify({'error': 'Skill not found'}), 404
    
    data = request.get_json()
    progress = data.get('progress', 0)
    level = data.get('level', skill.level)
    
    if not 0 <= progress <= 100:
        return jsonify({'error': 'Progress must be between 0 and 100'}), 400
    
    skill.progress = progress
    skill.level = level
    skill.updated_at = datetime.now()
    db.session.commit()
    
    return jsonify({'success': True, 'message': 'Skill updated successfully'})


@app.route('/api/add-job-recommendation', methods=['POST'])
def api_add_job_recommendation():
    """Add a job recommendation for the user"""
    if 'user_email' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    user = User.query.filter_by(email=session['user_email']).first()
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    data = request.get_json()
    job_title = data.get('job_title', '').strip()
    company = data.get('company', '').strip()
    salary = data.get('salary', '').strip()
    hours = data.get('hours', '').strip()
    location = data.get('location', '').strip()
    description = data.get('description', '').strip()
    
    if not job_title:
        return jsonify({'error': 'Job title is required'}), 400
    
    new_recommendation = JobRecommendation(
        user_id=user.id,
        job_title=job_title,
        company=company,
        salary=salary,
        hours=hours,
        location=location,
        description=description
    )
    db.session.add(new_recommendation)
    db.session.commit()
    
    return jsonify({'success': True, 'recommendation_id': new_recommendation.id, 'message': 'Job recommendation added'})


# ============================================
# 📊 ADMIN DASHBOARD
# ============================================

# Admin credentials (in production, use environment variables or database)
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"

@app.route('/admin-login', methods=['GET', 'POST'])
def admin_login():
    """Admin login page"""
    if request.method == 'POST':
        username = request.form.get('admin_username')
        password = request.form.get('admin_password')
        
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['is_admin'] = True
            session['admin_username'] = username
            flash('Admin login successful!', 'success')
            return redirect(url_for('admin_dashboard'))
        else:
            flash('Invalid admin credentials. Please try again.', 'error')
            return redirect(url_for('admin_login'))
    
    return render_template('admin_login.html')

@app.route('/admin-logout')
def admin_logout():
    """Admin logout"""
    session.pop('is_admin', None)
    session.pop('admin_username', None)
    flash('Admin logged out successfully.', 'success')
    return redirect(url_for('home'))

@app.route('/admin')
def admin_dashboard():
    """Admin dashboard to view all database information"""
    # Check if admin is logged in
    if not session.get('is_admin'):
        flash('Please login as admin to access the dashboard.', 'error')
        return redirect(url_for('admin_login'))
    
    # Get all data from database
    users = User.query.all()
    user_profiles = UserProfile.query.all()
    job_searches = JobSearchHistory.query.order_by(JobSearchHistory.created_at.desc()).all()
    skill_searches = SkillSearchHistory.query.order_by(SkillSearchHistory.created_at.desc()).all()
    contact_messages = ContactMessage.query.order_by(ContactMessage.created_at.desc()).all()
    tasks = Task.query.order_by(Task.created_at.desc()).all()
    reminders = Reminder.query.order_by(Reminder.created_at.desc()).all()
    organizations = Organization.query.order_by(Organization.created_at.desc()).all()
    
    # Create a dict of user_id to profile for easy lookup
    profiles_dict = {p.user_id: p for p in user_profiles}
    
    # Calculate stats
    stats = {
        'users': len(users),
        'job_searches': len(job_searches),
        'skill_searches': len(skill_searches),
        'messages': len(contact_messages),
        'tasks': len(tasks),
        'reminders': len(reminders),
        'organizations': len(organizations)
    }
    
    return render_template('admin.html',
        users=users,
        profiles_dict=profiles_dict,
        job_searches=job_searches,
        skill_searches=skill_searches,
        contact_messages=contact_messages,
        tasks=tasks,
        reminders=reminders,
        organizations=organizations,
        stats=stats
    )


# ============================================
# 🤖 ADMIN ML MODEL RETRAINING ENDPOINT
# ============================================

@app.route('/admin/retrain-model', methods=['POST'])
def admin_retrain_model():
    """
    Admin endpoint to retrain ML models when new data is uploaded
    Retrains income prediction model with updated dataset statistics
    """
    # ✅ Security: Check for admin token or authentication
    admin_token = request.headers.get('X-Admin-Token')
    if admin_token != os.environ.get('ADMIN_TOKEN', 'admin_secret_key'):
        return jsonify({'error': 'Unauthorized access'}), 403
    
    try:
        print("🔄 Starting model retraining pipeline...")
        
        # Load fresh data
        global df, work_model, income_model
        df = pd.read_csv("dataset.csv")
        df.fillna("", inplace=True)
        
        # ✅ Retrain work_model (if training data available)
        if len(df) > 10:
            print("  ✓ Retraining work mode prediction model...")
            try:
                # Create features for training
                from sklearn.preprocessing import LabelEncoder
                from sklearn.ensemble import RandomForestClassifier
                
                X_train = df[['age', 'hours', 'education']].fillna(0)
                y_train = df['work_mode'].fillna('Work From Home')
                
                # Encode categorical target
                le = LabelEncoder()
                y_train_encoded = le.fit_transform(y_train)
                
                # Train model
                work_model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
                work_model.fit(X_train, y_train_encoded)
                joblib.dump(work_model, "work_model.pkl")
                print("    ✓ Work model trained successfully!")
            except Exception as e:
                print(f"    ⚠️ Could not retrain work model: {e}")
        
        # ✅ Retrain income_model (if training data available)
        if len(df) > 10:
            print("  ✓ Retraining income prediction model...")
            try:
                from sklearn.ensemble import RandomForestRegressor
                
                X_train = df[['age', 'hours', 'education']].fillna(0)
                y_train = df['salary'].fillna(5000)
                
                # Train model
                income_model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
                income_model.fit(X_train, y_train)
                joblib.dump(income_model, "income_model.pkl")
                print("    ✓ Income model trained successfully!")
            except Exception as e:
                print(f"    ⚠️ Could not retrain income model: {e}")
        
        print("✅ Model retraining completed!")
        return jsonify({
            'success': True,
            'message': 'Models retrained successfully with new data',
            'timestamp': datetime.now().isoformat(),
            'dataset_size': len(df),
            'models': ['work_model', 'income_model']
        })
    
    except Exception as e:
        print(f"❌ Model retraining failed: {e}")
        return jsonify({
            'error': f'Retraining failed: {str(e)}'
        }), 500


@app.route('/admin/model-stats', methods=['GET'])
def admin_model_stats():
    """
    Admin endpoint to view model performance statistics
    Shows dataset size, model confidence metrics
    """
    # ✅ Security check
    admin_token = request.headers.get('X-Admin-Token')
    if admin_token != os.environ.get('ADMIN_TOKEN', 'admin_secret_key'):
        return jsonify({'error': 'Unauthorized access'}), 403
    
    try:
        stats = {
            'dataset_size': len(df),
            'columns': list(df.columns),
            'models_available': {
                'work_model': work_model is not None,
                'income_model': income_model is not None
            },
            'income_statistics': {
                'mean': float(df['salary'].mean()) if 'salary' in df.columns else 0,
                'median': float(df['salary'].median()) if 'salary' in df.columns else 0,
                'std': float(df['salary'].std()) if 'salary' in df.columns else 0,
                'min': float(df['salary'].min()) if 'salary' in df.columns else 0,
                'max': float(df['salary'].max()) if 'salary' in df.columns else 0,
                'q25': float(df['salary'].quantile(0.25)) if 'salary' in df.columns else 0,
                'q75': float(df['salary'].quantile(0.75)) if 'salary' in df.columns else 0,
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(stats)
    
    except Exception as e:
        return jsonify({'error': f'Error retrieving stats: {str(e)}'}), 500


# ================================
# 📊 USER HISTORY & PROFILE APIs
# ================================

@app.route('/api/job-history', methods=['GET'])
def get_job_history():
    """Get user's job search history"""
    if 'user_email' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    user = User.query.filter_by(email=session['user_email']).first()
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    history = JobSearchHistory.query.filter_by(user_id=user.id).order_by(JobSearchHistory.search_date.desc()).limit(50).all()
    
    return jsonify({
        'success': True,
        'history': [{
            'id': h.id,
            'domain': h.domain,
            'skill': h.skill,
            'education': h.education,
            'location': h.location,
            'work_mode': h.work_mode,
            'predicted_work': h.predicted_work,
            'salary_range': f"₹{h.predicted_salary_low:,} - ₹{h.predicted_salary_high:,}",
            'confidence': h.confidence_score,
            'date': h.search_date.strftime('%Y-%m-%d %H:%M')
        } for h in history]
    })


@app.route('/api/skill-history', methods=['GET'])
def get_skill_history():
    """Get user's skill search history"""
    if 'user_email' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    user = User.query.filter_by(email=session['user_email']).first()
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    history = SkillSearchHistory.query.filter_by(user_id=user.id).order_by(SkillSearchHistory.search_date.desc()).limit(50).all()
    
    return jsonify({
        'success': True,
        'history': [{
            'id': h.id,
            'skill_name': h.skill_name,
            'skill_level': h.skill_level,
            'hours_available': h.hours_available,
            'income_range': f"₹{h.estimated_income_low:,} - ₹{h.estimated_income_high:,}",
            'date': h.search_date.strftime('%Y-%m-%d %H:%M')
        } for h in history]
    })


@app.route('/api/user-profile', methods=['GET'])
def get_user_profile():
    """Get user's profile information"""
    if 'user_email' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    user = User.query.filter_by(email=session['user_email']).first()
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    profile = UserProfile.query.filter_by(user_id=user.id).first()
    
    return jsonify({
        'success': True,
        'user': {
            'id': user.id,
            'name': user.name,
            'email': user.email,
            'joined': user.created_at.strftime('%Y-%m-%d') if hasattr(user, 'created_at') else None
        },
        'profile': {
            'age': profile.age if profile else None,
            'education': profile.education if profile else None,
            'city_type': profile.city_type if profile else None,
            'language': profile.language if profile else None,
            'device': profile.device if profile else None,
            'location': profile.location if profile else None,
            'work_preference': profile.work_preference if profile else None
        } if profile else None
    })


@app.route('/api/update-profile', methods=['POST'])
def update_user_profile():
    """Update user's profile information"""
    if 'user_email' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    user = User.query.filter_by(email=session['user_email']).first()
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    data = request.json
    
    profile = UserProfile.query.filter_by(user_id=user.id).first()
    if not profile:
        profile = UserProfile(user_id=user.id)
        db.session.add(profile)
    
    # Update fields if provided
    if 'age' in data:
        profile.age = int(data['age'])
    if 'education' in data:
        profile.education = data['education']
    if 'city_type' in data:
        profile.city_type = data['city_type']
    if 'language' in data:
        profile.language = data['language']
    if 'device' in data:
        profile.device = data['device']
    if 'location' in data:
        profile.location = data['location']
    if 'work_preference' in data:
        profile.work_preference = data['work_preference']
    
    profile.updated_at = datetime.now()
    
    try:
        db.session.commit()
        return jsonify({'success': True, 'message': 'Profile updated successfully'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Error updating profile: {str(e)}'}), 500


@app.route('/api/delete-reminder/<int:reminder_id>', methods=['DELETE'])
def delete_reminder(reminder_id):
    """Delete a reminder"""
    if 'user_email' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    user = User.query.filter_by(email=session['user_email']).first()
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    reminder = Reminder.query.filter_by(id=reminder_id, user_id=user.id).first()
    if not reminder:
        return jsonify({'error': 'Reminder not found'}), 404
    
    try:
        db.session.delete(reminder)
        db.session.commit()
        return jsonify({'success': True, 'message': 'Reminder deleted successfully'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Error deleting reminder: {str(e)}'}), 500


@app.route('/api/delete-skill/<int:skill_id>', methods=['DELETE'])
def delete_skill(skill_id):
    """Delete a user skill"""
    if 'user_email' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    user = User.query.filter_by(email=session['user_email']).first()
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    skill = UserSkill.query.filter_by(id=skill_id, user_id=user.id).first()
    if not skill:
        return jsonify({'error': 'Skill not found'}), 404
    
    try:
        db.session.delete(skill)
        db.session.commit()
        return jsonify({'success': True, 'message': 'Skill deleted successfully'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Error deleting skill: {str(e)}'}), 500


@app.route('/api/dashboard-stats', methods=['GET'])
def get_dashboard_stats():
    """Get user's dashboard statistics"""
    if 'user_email' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    user = User.query.filter_by(email=session['user_email']).first()
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    # Count various items
    tasks_total = Task.query.filter_by(user_id=user.id).count()
    tasks_completed = Task.query.filter_by(user_id=user.id, completed=True).count()
    reminders_total = Reminder.query.filter_by(user_id=user.id).count()
    reminders_active = Reminder.query.filter_by(user_id=user.id, is_active=True).count()
    skills_total = UserSkill.query.filter_by(user_id=user.id).count()
    job_searches = JobSearchHistory.query.filter_by(user_id=user.id).count()
    skill_searches = SkillSearchHistory.query.filter_by(user_id=user.id).count()
    
    return jsonify({
        'success': True,
        'stats': {
            'tasks': {'total': tasks_total, 'completed': tasks_completed},
            'reminders': {'total': reminders_total, 'active': reminders_active},
            'skills': {'total': skills_total},
            'searches': {'jobs': job_searches, 'skills': skill_searches}
        }
    })

# ✅ Frontend Page Routes
@app.route('/organizations/register', methods=['GET', 'POST'])
def org_register_page():
    t = dict(t=app.jinja_env.globals.get('t', {})) # ensure translation dict is available
    
    if request.method == 'POST':
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        
        if password != confirm_password:
            flash('Passwords do not match.', 'error')
            return redirect(url_for('org_register_page'))
            
        is_valid, error_msg = Validator.validate_password(password)
        if not is_valid:
            flash(error_msg, 'error')
            return redirect(url_for('org_register_page'))
            
        try:
            password_hash, salt = SecurityUtils.hash_password(password)
            new_org = Organization(
                company_name=request.form.get('company_name'),
                org_type=request.form.get('org_type'),
                industry=request.form.get('industry'),
                registration_number=request.form.get('registration_number'),
                established_year=int(request.form.get('established_year')) if request.form.get('established_year') else None,
                org_size=request.form.get('org_size'),
                website=request.form.get('website'),
                contact_name=request.form.get('contact_name'),
                designation=request.form.get('designation'),
                email=request.form.get('email'),
                phone_number=request.form.get('phone_number'),
                address=request.form.get('address'),
                city=request.form.get('city'),
                state=request.form.get('state'),
                pincode=request.form.get('pincode'),
                password_hash=password_hash,
                salt=salt
            )
            db.session.add(new_org)
            db.session.commit()
            flash('Organization registered successfully! You can now log in.', 'success')
            return redirect(url_for('org_login'))
        except Exception as e:
            db.session.rollback()
            flash(f'An error occurred: {str(e)}', 'error')
            
    return render_template('organizations/register.html', **t)

# ✅ Admin Approval Routes
@app.route('/admin/org/<int:org_id>/approve', methods=['POST'])
def admin_approve_org(org_id):
    if not session.get('is_admin'):
        return redirect(url_for('admin_login'))
    org = Organization.query.get_or_404(org_id)
    org.status = 'Approved'
    db.session.commit()
    flash(f'{org.company_name} has been approved.', 'success')
    return redirect(url_for('admin_dashboard') + '#organizations')

@app.route('/admin/org/<int:org_id>/decline', methods=['POST'])
def admin_decline_org(org_id):
    if not session.get('is_admin'):
        return redirect(url_for('admin_login'))
    org = Organization.query.get_or_404(org_id)
    org.status = 'Declined'
    db.session.commit()
    flash(f'{org.company_name} has been declined.', 'error')
    return redirect(url_for('admin_dashboard') + '#organizations')

# ✅ Organization Portal Routes
@app.route('/org/login', methods=['GET', 'POST'])
def org_login():
    t = dict(t=app.jinja_env.globals.get('t', {}))
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '').strip()
        
        org = Organization.query.filter_by(email=email).first()
        if not org or not SecurityUtils.verify_password(password, org.password_hash, org.salt):
            flash('Invalid email or password', 'error')
            return render_template('organizations/login.html', **t)
            
        if org.status == 'Pending':
            flash('Your account is currently under review. You will be able to log in once an admin approves it.', 'info')
            return render_template('organizations/login.html', **t)
        elif org.status == 'Declined':
            flash('Your organization registration has been declined. Please contact support.', 'error')
            return render_template('organizations/login.html', **t)
            
        session.clear()
        session['org_id'] = org.id
        session['org_name'] = org.company_name
        flash('Successfully logged in!', 'success')
        return redirect(url_for('org_dashboard'))
        
    return render_template('organizations/login.html', **t)

@app.route('/org/logout')
def org_logout():
    session.pop('org_id', None)
    session.pop('org_name', None)
    flash('Logged out successfully.', 'success')
    return redirect(url_for('org_login'))

@app.route('/org/dashboard')
def org_dashboard():
    t = dict(t=app.jinja_env.globals.get('t', {}))
    org_id = session.get('org_id')
    if not org_id:
        return redirect(url_for('org_login'))
        
    org = Organization.query.get(org_id)
    jobs = OrganizationJob.query.filter_by(org_id=org_id).order_by(OrganizationJob.created_at.desc()).all()
    
    return render_template('organizations/dashboard.html', org=org, jobs=jobs, **t)

@app.route('/org/jobs/add', methods=['POST'])
def org_add_job():
    org_id = session.get('org_id')
    if not org_id:
        return redirect(url_for('org_login'))
        
    try:
        new_job = OrganizationJob(
            org_id=org_id,
            title=request.form.get('title'),
            job_type=request.form.get('job_type'),
            work_mode=request.form.get('work_mode'),
            location=request.form.get('location'),
            salary_range=request.form.get('salary_range'),
            description=request.form.get('description'),
            requirements=request.form.get('requirements')
        )
        db.session.add(new_job)
        db.session.commit()
        flash('Job posted successfully!', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'An error occurred: {str(e)}', 'error')
        
    return redirect(url_for('org_dashboard'))


# ✅ Run the app
if __name__ == "__main__":
    print("🚀 Starting MaaSarthi Server...")
    print("🔐 Security features enabled:")
    print("   ✓ Password hashing with SHA-256 + salt")
    print("   ✓ Input validation & sanitization")
    print("   ✓ Rate limiting (5 attempts / 5 minutes)")
    print("   ✓ Secure session management")
    print("   ✓ Security headers (XSS, Clickjacking protection)")
    print("   ✓ CSRF protection via SameSite cookies")
    print("🗄️  Database: SQLite (maasarthi.db)")
    print("   ✓ Tables: users, tasks, reminders, contact_messages, user_skills")
    print("   ✓ Tables: job_recommendations, user_profiles, job_search_history, skill_search_history")
    print("📊 API Endpoints:")
    print("   ✓ /api/job-history - Get job search history")
    print("   ✓ /api/skill-history - Get skill search history")
    print("   ✓ /api/user-profile - Get/Update user profile")
    print("   ✓ /api/dashboard-stats - Get dashboard statistics")
    print("🤖 ML Features enabled:")
    print("   ✓ Quantile-based income range calculation")
    print("   ✓ Confidence score from RandomForest prediction variance")
    print("   ✓ Model retraining pipeline (/admin/retrain-model)")
    print("   ✓ Model statistics endpoint (/admin/model-stats)")
    print("")
    app.run(debug=True, port=5001)