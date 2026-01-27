from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import joblib
import pandas as pd
from flask import send_from_directory
import os

from pathlib import Path
from flask import Flask, request, jsonify, render_template
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity






# ✅ Safe model loader
def safe_load_model(path: str):
    """Load a joblib model safely. If missing, return None instead of crashing the app."""
    try:
        if Path(path).exists():
            return joblib.load(path)
    except Exception as e:
        print(f"⚠️ Could not load model {path}: {e}")
    return None

# ✅ Initialize Flask app
app = Flask(__name__)
app.secret_key = "maasarthi_secret_key_123"

# ✅ Simple in-memory user storage (for development)
users_db = {}

# ✅ Login required decorator
from functools import wraps

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_email' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# ✅ Load dataset.csv (FREE search system)
df = pd.read_csv("dataset.csv")
df.fillna("", inplace=True)

# Convert each row into one text string
rows_text = df.astype(str).agg(" | ".join, axis=1).tolist()

# Create vectorizer
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(rows_text)


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
            "Security","Music","Electrician","Handicraft","Fitness","IT"
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
            "सिक्योरिटी","म्यूजिक","इलेक्ट्रीशियन","हस्तकला","फिटनेस","आईटी"
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
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()
        
        # Validate input
        if not email or not password:
            return render_template('login.html', error='Email and password are required')
        
        # Check if user exists and password is correct
        if email in users_db and users_db[email]['password'] == password:
            session['user_email'] = email
            session['user_name'] = users_db[email]['name']
            return redirect('/dashboard')
        else:
            return render_template('login.html', error='Invalid email or password')
    
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        fullname = request.form.get('fullname', '').strip()
        email = request.form.get('email', '').strip()
        phone = request.form.get('phone', '').strip()
        password = request.form.get('password', '').strip()
        confirm_password = request.form.get('confirm_password', '').strip()
        
        # Validation
        if not all([fullname, email, phone, password, confirm_password]):
            return render_template('signup.html', error='All fields are required')
        
        if len(password) < 6:
            return render_template('signup.html', error='Password must be at least 6 characters')
        
        if password != confirm_password:
            return render_template('signup.html', error='Passwords do not match')
        
        if email in users_db:
            return render_template('signup.html', error='Email already registered')
        
        # Register new user
        users_db[email] = {
            'name': fullname,
            'phone': phone,
            'password': password
        }
        
        return redirect('/login?success=Account+created+successfully')
    
    return render_template('signup.html')

@app.route('/forgot-password')
def forgot_password():
    return render_template('forgot_password.html')

@app.route('/dashboard')
def dashboard():
    if 'user_email' not in session:
        return redirect('/login')
    return render_template('dashboard.html', user_name=session.get('user_name', 'User'))

@app.route('/logout')
def logout():
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
    return render_template("home.html", t=t)

# ✅ Skills page

# ✅ Skills page
@app.route("/skills")
@login_required
def skills():
    t = get_text()
    return render_template("skill_form.html", t=t)

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
        df = pd.read_csv("dataset.csv")

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

    return render_template("result.html", t=t, work=f"{skill} Training", low=low, high=high, skills=skills, links=links)

# ✅ Find jobs nearby
@app.route("/find-jobs-nearby")
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
    return render_template("form.html", t=t)

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
    else:
        work_pred = work_model.predict(X)[0]
        income_pred = int(income_model.predict(X)[0])

    low = (income_pred // 1000) * 1000
    high = low + 1000

    suggested_skills = ["Improve your skill", "Create Portfolio", "Market yourself online"]
    links = [("YouTube", "https://www.youtube.com"), ("Google", "https://www.google.com")]

    return render_template("result.html", t=t, work=work_pred, low=low, high=high, skills=suggested_skills, links=links)

# ✅ Run the app
if __name__ == "__main__":
    app.run(debug=True, port=5001)