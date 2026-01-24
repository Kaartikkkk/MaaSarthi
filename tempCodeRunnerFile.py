from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import joblib
import pandas as pd
from flask import send_from_directory
import os
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path

# ✅ Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
        "get_rec": "✅ सुझाव देखें",
        "train_instead": "📚 स्किल ट्रेनिंग करें",
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
        "get_training": "🎯 ट्रेनिंग प्लान देखें",
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

# ✅ Helper function to get language text
def get_text():
    lang = session.get("lang", "en")
    return TEXT.get(lang, TEXT["en"])

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

# ✅ CHAT ROUTE - UNIFIED (Only one /chat route)
@app.route("/chat", methods=["POST"])
def chat():
    """
    Smart Chat Assistant:
    - Uses OpenAI GPT for intelligent responses
    - Uses dataset.csv for job/skill suggestions
    - Works for both Hindi and English
    """
    t = get_text()
    user_msg = request.json.get("message", "").strip()

    if not user_msg:
        return jsonify({"reply": "Please type something 😊"})

    msg_lower = user_msg.lower()
    lang = session.get("lang", "en")

    # ✅ Load dataset for smart suggestions
    try:
        df = pd.read_csv("dataset.csv")
    except:
        df = None

    # ✅ Basic Intent Detection (Fast responses without API)
    if any(word in msg_lower for word in ["job", "work", "earning", "income", "salary"]):
        if df is not None:
            top_jobs = df["job"].value_counts().head(5).index.tolist()
            if lang == "hi":
                return jsonify({"reply": "✅ आपके लिए कुछ अच्छे काम:\n👉 " + "\n👉 ".join(top_jobs)})
            return jsonify({"reply": "✅ Here are some good job options:\n👉 " + "\n👉 ".join(top_jobs)})

    if any(word in msg_lower for word in ["skill", "learn", "training", "course"]):
        if df is not None:
            top_skills = df["skill"].value_counts().head(5).index.tolist()
            if lang == "hi":
                return jsonify({"reply": "📚 सीखने के लिए सबसे अच्छी स्किल्स:\n👉 " + "\n👉 ".join(top_skills)})
            return jsonify({"reply": "📚 Best skills to learn:\n👉 " + "\n👉 ".join(top_skills)})

    if any(word in msg_lower for word in ["roadmap", "plan", "steps"]):
        if lang == "hi":
            return jsonify({"reply": 
                "✅ ट्रेनिंग रोडमैप:\n"
                "1️⃣ बेसिक्स सीखें\n"
                "2️⃣ रोज़ प्रैक्टिस करें\n"
                "3️⃣ छोटे प्रोजेक्ट बनाएं\n"
                "4️⃣ पोर्टफोलियो तैयार करें\n"
                "5️⃣ जॉब/फ्रीलांसिंग के लिए अप्लाई करें"
            })
        return jsonify({"reply":
            "✅ Training Roadmap:\n"
            "1️⃣ Learn basics\n"
            "2️⃣ Practice daily\n"
            "3️⃣ Build small projects\n"
            "4️⃣ Create portfolio\n"
            "5️⃣ Apply for jobs or freelancing"
        })

    if any(word in msg_lower for word in ["hindi", "भाषा", "हिंदी"]):
        return jsonify({"reply": "✅ आप भाषा बदलने के लिए Home page से हिंदी चुन सकते हैं।"})

    # ✅ Use OpenAI for complex queries
    system_prompt = """
You are MaaSarthi Assistant.
You help mothers find jobs, earning options, and skill training plans.
Reply in Hindi if user writes in Hindi, otherwise reply in English.
If user asks anything unrelated, still respond helpfully.
Keep replies short and friendly.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg}
            ],
            max_tokens=200
        )
        reply = response.choices[0].message.content.strip()
        return jsonify({"reply": reply})

    except Exception as e:
        print("❌ Chat error:", e)
        
        # Fallback response if API fails
        if lang == "hi":
            return jsonify({"reply":
                "🤖 मैं MaaSarthi Assistant हूँ ✅\n"
                "आप मुझसे पूछ सकते हैं:\n"
                "👉 कौन सी जॉब सबसे अच्छी है?\n"
                "👉 कितनी कमाई हो सकती है?\n"
                "👉 कौन सी स्किल सीखनी चाहिए?\n"
                "👉 स्किल रोडमैप क्या होगा?\n\n"
                "आपका सवाल थोड़ा और स्पष्ट करें 😊"
            })
        
        return jsonify({"reply":
            "🤖 I am MaaSarthi Assistant ✅\n"
            "You can ask me:\n"
            "👉 Which job is best for me?\n"
            "👉 How much can I earn?\n"
            "👉 What skill should I learn?\n"
            "👉 Give me a training roadmap\n\n"
            "Please ask in a little more detail 😊"
        })

# ✅ Skills page
@app.route("/skills")
def skills():
    t = get_text()
    return render_template("skill_form.html", t=t)

# ✅ Skills result
@app.route("/skills-result", methods=["POST"])
@app.route("/skills_result", methods=["POST"])
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
def jobs():
    t = get_text()
    return render_template("form.html", t=t)

# ✅ Predict job recommendation
@app.route("/predict", methods=["POST"])
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
    app.run(debug=True)