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
# 🔐 SECURITY & VALIDATION UTILITIES
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
            # Store the URL the user wanted to access
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
work_model = safe_load_model("work_model.pkl")
income_model = safe_load_model("income_model.pkl")

# ✅ Language Dictionary (EN + HI)
TEXT = {
    "en": {
        "education_list": {
            "No Formal": "No Formal",
            "8th": "8th",
            "10th": "10th",
            "12th": "12th",
            "Diploma": "Diploma",
            "UG": "UG",
            "PG": "PG"
        },
        "city_type_list": {
            "Urban": "Urban",
            "Semi-Urban": "Semi-Urban",
            "Rural": "Rural"
        },
        "language_list": {
            "Hindi": "Hindi",
            "English": "English",
            "Both": "Both"
        },
        "device_list": {
            "Mobile": "Mobile",
            "Laptop": "Laptop",
            "Both": "Both"
        },
        "work_mode_list": {
            "Work From Home": "Work From Home",
            "Hybrid": "Hybrid",
            "Offline Local": "Offline Local"
        },
        "domain_list": [
            "Cooking","Baking","Teaching","Commerce","Law","Cleaning","Gardening",
            "Data Entry","Social Media","E-Commerce","Clothing","Beauty","Banking",
            "Security","Music","Electrician","Handicraft","Fitness","IT",
            "Writing","Translation","Graphic Design","Photography","Video Editing",
            "Web Development","Customer Support","Virtual Assistant","Accounting",
            "Healthcare","Nursing","Yoga","Meditation","Tutoring","Art & Craft",
            "Interior Design","Event Planning","Catering","Tailoring","Embroidery",
            "Mehndi","Makeup Artist","Hair Styling","Nutrition","Consulting","Other"
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
        "kids": "Kids",
        "hours": "Hours Available Per Day",
        "domain": "Domain",
        "main_skill": "Main Skill",
        "education": "Education",
        "city_type": "City Type",
        "language": "Language",
        "device": "Device",
        "work_mode": "Work Mode",
        "get_rec": "✅ Get Recommendation",
        "train_instead": "📚 Train a Skill Instead",
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
            "No Formal": "कोई औपचारिक शिक्षा नहीं",
            "8th": "8वीं",
            "10th": "10वीं",
            "12th": "12वीं",
            "Diploma": "डिप्लोमा",
            "UG": "स्नातक (UG)",
            "PG": "स्नातकोत्तर (PG)"
        },
        "city_type_list": {
            "Urban": "शहरी",
            "Semi-Urban": "अर्ध-शहरी",
            "Rural": "ग्रामीण"
        },
        "language_list": {
            "Hindi": "हिन्दी",
            "English": "अंग्रेज़ी",
            "Both": "दोनों"
        },
        "device_list": {
            "Mobile": "मोबाइल",
            "Laptop": "लैपटॉप",
            "Both": "दोनों"
        },
        "work_mode_list": {
            "Work From Home": "घर से काम",
            "Hybrid": "हाइब्रिड",
            "Offline Local": "लोकल ऑफलाइन"
        },
        "domain_list": [
            "कुकिंग","बेकिंग","टीचिंग","कॉमर्स","कानून","सफाई","बागवानी",
            "डेटा एंट्री","सोशल मीडिया","ई-कॉमर्स","कपड़े","ब्यूटी","बैंकिंग",
            "सिक्योरिटी","म्यूजिक","इलेक्ट्रीशियन","हस्तकला","फिटनेस","आईटी",
            "लेखन","अनुवाद","ग्राफिक डिज़ाइन","फोटोग्राफी","वीडियो एडिटिंग",
            "वेब डेवलपमेंट","कस्टमर सपोर्ट","वर्चुअल असिस्टेंट","अकाउंटिंग",
            "हेल्थकेयर","नर्सिंग","योगा","मेडिटेशन","ट्यूटरिंग","आर्ट एंड क्राफ्ट",
            "इंटीरियर डिज़ाइन","इवेंट प्लानिंग","केटरिंग","टेलरिंग","कढ़ाई",
            "मेहंदी","मेकअप आर्टिस्ट","हेयर स्टाइलिंग","न्यूट्रिशन","कंसल्टिंग","अन्य"
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
        "kids": "बच्चे",
        "hours": "दिन में उपलब्ध समय (घंटे)",
        "domain": "क्षेत्र (डोमेन)",
        "main_skill": "मुख्य स्किल",
        "education": "शिक्षा",
        "city_type": "शहर का प्रकार",
        "language": "भाषा",
        "device": "डिवाइस",
        "work_mode": "काम का तरीका",
        "get_rec": "सुझाव देखें",
        "train_instead": " स्किल ट्रेनिंग करें",
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

# ✅ Predict job recommendation
@app.route("/predict", methods=["POST"])
@login_required
def predict():
    t = get_text()

    age = int(request.form.get("age", 0))
    kids = int(request.form.get("kids", 0))
    hours = int(request.form.get("hours", 0))
    domain = request.form.get("domain", "Cooking")
    skill = request.form.get("skill", "Cooking")
    education = request.form.get("education", "10th")
    city_type = request.form.get("city_type", "Urban")

    location = request.form.get("location", "").strip()
    if location:
        session["location"] = location

    language = request.form.get("language", "Hindi")
    device = request.form.get("device", "Mobile")
    work_mode = request.form.get("work_mode", "Work From Home")

    X = pd.DataFrame([{
        "age": age,
        "kids": kids,
        "hours": hours,
        "domain": domain,
        "skill": skill,
        "education": education,
        "city_type": city_type,
        "language": language,
        "device": device,
        "work_mode": work_mode
    }])

    if work_model is None or income_model is None:
        work_pred = "Work From Home"
        income_pred = 5000
        confidence_score = 0.65
    else:
        work_pred = work_model.predict(X)[0]
        income_pred = int(income_model.predict(X)[0])
        
        # ✅ Get confidence score from RandomForestClassifier/Regressor
        try:
            if hasattr(income_model, 'predict_proba'):
                # For classifier: get max probability
                confidence_score = float(np.max(income_model.predict_proba(X)[0]))
            elif hasattr(income_model, '_estimators'):
                # For RandomForestRegressor: calculate mean prediction variance
                predictions = np.array([tree.predict(X)[0] for tree in income_model.estimators_])
                variance = np.var(predictions)
                # Normalize variance to confidence score (0-1)
                confidence_score = float(1.0 / (1.0 + variance / 1000))
                confidence_score = min(0.99, max(0.5, confidence_score))
            else:
                confidence_score = 0.80
        except Exception as e:
            print(f"Error calculating confidence: {e}")
            confidence_score = 0.75
    
    # ✅ QUANTILE-BASED INCOME RANGE (from dataset statistics)
    try:
        # ✅ Load cached dataset
        global df
        df, _, _, _ = load_dataset()
        
        if df is None or df.empty:
            raise Exception("Dataset not loaded")
        
        # Get income statistics for similar domain from dataset
        domain_data = df[df['Domain'].str.contains(domain, case=False, na=False)]
        
        if len(domain_data) > 0 and 'Salary' in df.columns:
            # Extract salary values and calculate quantiles
            salary_values = []
            for salary_str in domain_data['Salary'].dropna():
                try:
                    # Extract numeric value from salary string
                    import re
                    numbers = re.findall(r'\d+', str(salary_str))
                    if numbers:
                        salary_values.append(int(numbers[0]) * 1000)  # Convert to full amount
                except:
                    pass
            
            if salary_values:
                # Use 25th and 75th percentiles for range
                low = int(np.percentile(salary_values, 25))
                high = int(np.percentile(salary_values, 75))
            else:
                # Fallback: use model prediction with ±30% range
                low = int(income_pred * 0.70)
                high = int(income_pred * 1.30)
        else:
            # Default range based on model prediction
            low = int(income_pred * 0.70)
            high = int(income_pred * 1.30)
    except Exception as e:
        print(f"Error calculating quantile range: {e}")
        # Fallback to original fixed range
        low = (income_pred // 1000) * 1000
        high = low + 1000

    # ✅ Get multiple job recommendations from dataset
    job_recommendations = []
    try:
        # ✅ Load cached dataset if not already loaded
        if df is None or df.empty:
            df, _, _, _ = load_dataset()
        
        # Search for jobs matching the domain and work mode
        filtered_df = df[
            (df['Domain'].str.contains(domain, case=False, na=False)) |
            (df['Work Mode'].str.contains(work_mode, case=False, na=False))
        ]
        
        # Get top 5 jobs or all available
        top_jobs = filtered_df.head(5).to_dict('records')
        
        for idx, job in enumerate(top_jobs, 1):
            job_title = job.get('Job Title', 'Job Opportunity')
            job_recommendations.append({
                'id': idx,
                'title': job_title,
                'company': job.get('Company', 'MaaSarthi Partner'),
                'description': job.get('Description', 'Remote work opportunity'),
                'salary': job.get('Salary', f'₹{low} - ₹{high}'),
                'work_mode': job.get('Work Mode', work_mode),
                'location': job.get('Location', location or 'Remote'),
                'apply_links': {
                    'linkedin': f"https://www.linkedin.com/jobs/search/?keywords={job_title.replace(' ', '%20')}&location=India",
                    'naukri': f"https://www.naukri.com/{job_title.lower().replace(' ', '-')}-jobs",
                    'indeed': f"https://www.indeed.co.in/jobs?q={job_title.replace(' ', '+')}&l=India",
                    'internshala': f"https://internshala.com/jobs/{job_title.lower().replace(' ', '-')}-jobs"
                }
            })
    except Exception as e:
        print(f"Error fetching jobs: {e}")
        # Fallback: create sample recommendations with real job links
        search_term = domain.replace(' ', '%20')
        search_term_dash = domain.lower().replace(' ', '-')
        search_term_plus = domain.replace(' ', '+')
        
        job_recommendations = [
            {
                'id': 1,
                'title': f'{domain} - Remote Position',
                'company': 'MaaSarthi Partner',
                'description': f'Part-time {domain} work from home',
                'salary': f'₹{low} - ₹{high}',
                'work_mode': work_mode,
                'location': location or 'Remote',
                'apply_links': {
                    'linkedin': f"https://www.linkedin.com/jobs/search/?keywords={search_term}&location=India&f_WT=2",
                    'naukri': f"https://www.naukri.com/{search_term_dash}-jobs?wfhType=2",
                    'indeed': f"https://www.indeed.co.in/jobs?q={search_term_plus}&l=India&remotejob=032b3046-06a3-4876-8dfd-474eb5e7ed11",
                    'internshala': f"https://internshala.com/jobs/{search_term_dash}-jobs/work-from-home"
                }
            },
            {
                'id': 2,
                'title': f'{domain} Freelancer',
                'company': 'Independent',
                'description': f'Flexible {domain} freelancing opportunities',
                'salary': f'₹{low*2} - ₹{high*2}',
                'work_mode': 'Flexible',
                'location': 'Remote',
                'apply_links': {
                    'linkedin': f"https://www.linkedin.com/jobs/search/?keywords={search_term}%20freelance&location=India",
                    'naukri': f"https://www.naukri.com/freelance-{search_term_dash}-jobs",
                    'indeed': f"https://www.indeed.co.in/jobs?q=freelance+{search_term_plus}&l=India",
                    'internshala': f"https://internshala.com/jobs/{search_term_dash}-jobs"
                }
            },
            {
                'id': 3,
                'title': f'{domain} Expert',
                'company': 'MaaSarthi Network',
                'description': f'Become a {domain} expert and earn',
                'salary': f'₹{low*1.5} - ₹{high*1.5}',
                'work_mode': work_mode,
                'location': 'Remote',
                'apply_links': {
                    'linkedin': f"https://www.linkedin.com/jobs/search/?keywords=senior%20{search_term}&location=India",
                    'naukri': f"https://www.naukri.com/senior-{search_term_dash}-jobs",
                    'indeed': f"https://www.indeed.co.in/jobs?q=senior+{search_term_plus}&l=India",
                    'internshala': f"https://internshala.com/jobs/{search_term_dash}-jobs"
                }
            }
        ]

    suggested_skills = ["Improve your skill", "Create Portfolio", "Market yourself online"]
    links = [("YouTube", "https://www.youtube.com"), ("Google", "https://www.google.com")]

    # ✅ Convert confidence score to percentage and round
    confidence_percentage = int(round(confidence_score * 100))

    # ✅ Save job search to database
    if 'user_email' in session:
        try:
            user = User.query.filter_by(email=session['user_email']).first()
            if user:
                # Save to job search history
                job_search = JobSearchHistory(
                    user_id=user.id,
                    age=age,
                    education=education,
                    domain=domain,
                    skill=skill,
                    work_mode=work_mode,
                    location=location,
                    city_type=city_type,
                    hours=hours,
                    kids=kids,
                    language=language,
                    device=device,
                    predicted_job=work_pred,
                    predicted_salary_low=low,
                    predicted_salary_high=high,
                    confidence_score=confidence_score
                )
                db.session.add(job_search)
                
                # Save the top job recommendation
                if job_recommendations:
                    top_job = job_recommendations[0]
                    job_rec = JobRecommendation(
                        user_id=user.id,
                        job_title=top_job.get('title', work_pred),
                        company=top_job.get('company', 'MaaSarthi Partner'),
                        salary=f'₹{low} - ₹{high}',
                        hours=f'{hours} hours/week',
                        location=location or 'Remote',
                        description=top_job.get('description', '')
                    )
                    db.session.add(job_rec)
                
                # Update or create user profile with search data
                profile = UserProfile.query.filter_by(user_id=user.id).first()
                if not profile:
                    profile = UserProfile(user_id=user.id)
                    db.session.add(profile)
                
                profile.age = age
                profile.education = education
                profile.location = location
                profile.city_type = city_type
                profile.preferred_domain = domain
                profile.primary_skill = skill
                profile.work_mode_preference = work_mode
                profile.available_hours = hours
                profile.number_of_kids = kids
                profile.language_preference = language
                profile.device_type = device
                profile.profile_completed = True
                
                db.session.commit()
                print(f"✅ Job search saved for user {user.email}: {work_pred}")
        except Exception as e:
            print(f"❌ Error saving job search: {e}")
            db.session.rollback()

    return render_template(
        "result.html",
        t=t,
        work=work_pred,
        low=low,
        high=high,
        skills=suggested_skills,
        links=links,
        jobs=job_recommendations,
        confidence_score=confidence_percentage,  # ✅ Pass confidence percentage to template
        income_prediction=income_pred,  # ✅ Pass actual prediction
        domain=domain,  # ✅ Pass domain for context
        user=get_current_user()
    )

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
    
    # Create a dict of user_id to profile for easy lookup
    profiles_dict = {p.user_id: p for p in user_profiles}
    
    # Calculate stats
    stats = {
        'users': len(users),
        'job_searches': len(job_searches),
        'skill_searches': len(skill_searches),
        'messages': len(contact_messages),
        'tasks': len(tasks),
        'reminders': len(reminders)
    }
    
    return render_template('admin.html',
        users=users,
        profiles_dict=profiles_dict,
        job_searches=job_searches,
        skill_searches=skill_searches,
        contact_messages=contact_messages,
        tasks=tasks,
        reminders=reminders,
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
    app.run(debug=False, port=5001)