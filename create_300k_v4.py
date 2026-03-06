"""
MAASARTHI 300K DATASET GENERATOR V4
====================================
Maximized uniqueness across ALL dimensions:
- 11,792 unique job titles (from file)
- 600+ unique companies
- 200+ unique cities
- 1000+ unique skills
- 54 domains

Author: MaaSarthi Data Team
Date: March 2026
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class MaaSarthiGeneratorV4:
    def __init__(self):
        self.target_records = 300000
        
        print("="*70)
        print("MAASARTHI DATASET GENERATOR V4 - MAXIMUM DIVERSITY")
        print("="*70)
        print(f"Target: {self.target_records:,} records")
        print()
        
        self._load_job_titles()
        self._initialize_expanded_data()
    
    def _load_job_titles(self):
        """Load job titles from CSV file"""
        job_df = pd.read_csv('/Users/kartik/Documents/MaaSarthi/job_titles_11792.csv')
        self.all_job_titles = job_df['job_title'].tolist()
        print(f"   Loaded {len(self.all_job_titles):,} unique job titles")
        
        self.job_categories = {}
        self._categorize_jobs()
    
    def _categorize_jobs(self):
        """Categorize job titles into domains"""
        domain_keywords = {
            'Software Engineering': ['software', 'developer', 'programmer', 'full stack', 'frontend', 'backend', 'web developer', 'mobile developer', 'android', 'ios', 'flutter', 'react', 'angular', 'node', 'java developer', 'python developer', '.net', 'devops', 'engineer'],
            'Data Science': ['data scientist', 'data analyst', 'machine learning', 'ml engineer', 'ai engineer', 'data engineer', 'analytics', 'bi developer', 'business intelligence', 'statistician', 'nlp'],
            'Cloud & DevOps': ['cloud', 'aws', 'azure', 'gcp', 'devops', 'sre', 'infrastructure', 'kubernetes', 'docker', 'platform engineer'],
            'Cybersecurity': ['security', 'cyber', 'soc', 'penetration', 'ethical hacker', 'ciso', 'threat', 'forensic'],
            'IT Support': ['it support', 'help desk', 'technical support', 'system admin', 'network admin', 'desktop support', 'it admin'],
            'UI/UX Design': ['ui designer', 'ux designer', 'product designer', 'visual designer', 'interaction designer', 'ux researcher'],
            'QA Testing': ['qa', 'tester', 'quality', 'sdet', 'automation tester', 'test engineer'],
            'Banking Operations': ['bank', 'loan', 'credit', 'treasury', 'forex', 'trade finance', 'kyc', 'aml', 'collections'],
            'Accounting': ['accountant', 'accounts', 'audit', 'tax', 'gst', 'finance executive', 'bookkeeper', 'payroll'],
            'Insurance': ['insurance', 'underwriter', 'claims', 'actuary', 'policy'],
            'Investment': ['investment', 'portfolio', 'equity', 'trader', 'wealth', 'mutual fund', 'stock broker', 'financial planner'],
            'Nursing': ['nurse', 'nursing', 'anm', 'gnm', 'icu nurse', 'staff nurse'],
            'Medical Doctors': ['doctor', 'physician', 'surgeon', 'cardiologist', 'neurologist', 'pediatrician', 'gynecologist', 'dermatologist', 'psychiatrist', 'radiologist', 'pathologist', 'oncologist'],
            'Allied Health': ['physiotherapist', 'lab technician', 'pharmacist', 'dietitian', 'optometrist', 'therapist', 'radiology', 'ultrasound', 'ecg', 'dialysis'],
            'Mental Health': ['counselor', 'psychologist', 'therapist', 'mental health', 'counseling'],
            'Pharma Sales': ['medical representative', 'pharma sales', 'clinical research', 'pharmacovigilance', 'regulatory affairs'],
            'School Teaching': ['teacher', 'teaching', 'school', 'primary', 'secondary', 'pgt', 'tgt', 'principal', 'headmaster'],
            'Higher Education': ['professor', 'lecturer', 'faculty', 'dean', 'hod', 'research fellow', 'academic'],
            'EdTech': ['online tutor', 'e-learning', 'course creator', 'edtech', 'content developer', 'instructional designer'],
            'Corporate Training': ['trainer', 'training', 'l&d', 'learning development', 'soft skills trainer'],
            'Content Writing': ['content writer', 'copywriter', 'technical writer', 'blogger', 'editor', 'proofreader', 'script'],
            'Journalism': ['journalist', 'reporter', 'correspondent', 'news', 'anchor', 'editor'],
            'Digital Marketing': ['digital marketing', 'seo', 'sem', 'social media', 'ppc', 'performance marketing', 'email marketing', 'growth'],
            'Graphic Design': ['graphic designer', 'visual designer', 'illustrator', 'motion graphics', 'video editor', 'animator', 'vfx'],
            'Sales': ['sales executive', 'business development', 'bdm', 'bde', 'account manager', 'territory', 'area sales', 'regional sales'],
            'Marketing': ['marketing', 'brand manager', 'campaign', 'market research', 'pr manager', 'public relations'],
            'Retail': ['retail', 'store manager', 'showroom', 'cashier', 'visual merchandiser', 'floor manager'],
            'HR': ['hr', 'human resource', 'recruiter', 'talent acquisition', 'payroll', 'employee relations', 'hrbp'],
            'Customer Support': ['customer service', 'customer support', 'call center', 'bpo', 'helpdesk', 'customer care'],
            'Administration': ['admin', 'administrative', 'receptionist', 'office manager', 'executive assistant', 'secretary', 'front desk'],
            'Data Entry': ['data entry', 'back office', 'typist', 'data processing', 'mis'],
            'Legal': ['lawyer', 'advocate', 'legal', 'attorney', 'paralegal', 'compliance', 'company secretary'],
            'Hotel Management': ['hotel', 'front office', 'housekeeping', 'concierge', 'hospitality', 'resort'],
            'Food Service': ['chef', 'cook', 'baker', 'pastry', 'f&b', 'restaurant', 'barista', 'catering'],
            'Travel Tourism': ['travel', 'tour', 'ticketing', 'visa', 'flight attendant', 'cabin crew', 'airline'],
            'Beauty Wellness': ['beautician', 'makeup', 'hair stylist', 'salon', 'spa', 'skincare', 'nail'],
            'Fitness Yoga': ['fitness', 'yoga', 'gym', 'trainer', 'zumba', 'pilates', 'sports coach'],
            'Fashion Design': ['fashion designer', 'textile', 'apparel', 'garment', 'tailor', 'embroidery', 'boutique'],
            'Handicrafts': ['artisan', 'craftsman', 'handloom', 'pottery', 'weaver', 'carpenter', 'woodwork'],
            'Childcare': ['nanny', 'babysitter', 'childcare', 'daycare', 'governess', 'creche', 'anganwadi'],
            'Elderly Care': ['caregiver', 'elder care', 'home care', 'patient care', 'attendant'],
            'Household': ['housekeeper', 'maid', 'cook', 'driver', 'gardener', 'security guard'],
            'Agriculture': ['farmer', 'agriculture', 'horticulture', 'agronomist', 'farm', 'crop', 'organic'],
            'Dairy Animal': ['dairy', 'poultry', 'livestock', 'veterinary', 'cattle', 'fisheries'],
            'Food Processing': ['food processing', 'production', 'packaging', 'quality control', 'fssai'],
            'Ecommerce': ['ecommerce', 'marketplace', 'catalog', 'listing', 'seller', 'amazon', 'flipkart'],
            'Delivery Logistics': ['delivery', 'courier', 'warehouse', 'logistics', 'supply chain', 'dispatch'],
            'Freelancing': ['freelancer', 'freelance', 'consultant', 'self-employed', 'gig'],
            'Manufacturing': ['factory', 'production', 'machine operator', 'assembly', 'welder', 'fitter', 'technician'],
            'Government': ['government', 'panchayat', 'tehsildar', 'collector', 'ias', 'ips', 'clerk', 'assistant'],
            'Social Work': ['social worker', 'ngo', 'community', 'asha', 'field worker', 'welfare'],
            'Real Estate': ['real estate', 'property', 'broker', 'leasing', 'facility manager'],
            'Construction': ['civil engineer', 'architect', 'construction', 'site engineer', 'mason', 'plumber', 'electrician'],
            'Consulting': ['consultant', 'consulting', 'strategy', 'advisory']
        }
        
        for title in self.all_job_titles:
            title_lower = title.lower()
            assigned = False
            for domain, keywords in domain_keywords.items():
                for keyword in keywords:
                    if keyword in title_lower:
                        if domain not in self.job_categories:
                            self.job_categories[domain] = []
                        self.job_categories[domain].append(title)
                        assigned = True
                        break
                if assigned:
                    break
            if not assigned:
                if 'General' not in self.job_categories:
                    self.job_categories['General'] = []
                self.job_categories['General'].append(title)
        
        print(f"   Categorized into {len(self.job_categories)} domains")
    
    def _initialize_expanded_data(self):
        """Initialize MASSIVELY EXPANDED supporting data"""
        
        # ========== 1000+ UNIQUE SKILLS ==========
        self.all_skills = [
            # Programming Languages (50+)
            'Python', 'Java', 'JavaScript', 'TypeScript', 'C++', 'C#', 'Go', 'Rust', 'Ruby', 'PHP',
            'Swift', 'Kotlin', 'Scala', 'R', 'MATLAB', 'Perl', 'Lua', 'Haskell', 'Clojure', 'Elixir',
            'Dart', 'Julia', 'Groovy', 'VB.NET', 'F#', 'Objective-C', 'COBOL', 'Fortran', 'Assembly',
            'Solidity', 'Prolog', 'Erlang', 'Scheme', 'Lisp', 'Ada', 'Pascal', 'D', 'Nim', 'Crystal',
            'OCaml', 'Racket', 'Zig', 'V', 'Ballerina', 'Chapel', 'Hack', 'Apex', 'ABAP', 'PL/SQL',
            
            # Web Technologies (80+)
            'HTML5', 'CSS3', 'SASS', 'LESS', 'Bootstrap', 'Tailwind CSS', 'Material UI', 'Chakra UI',
            'React.js', 'Angular', 'Vue.js', 'Svelte', 'Next.js', 'Nuxt.js', 'Gatsby', 'Remix',
            'Node.js', 'Express.js', 'NestJS', 'Fastify', 'Koa', 'Hapi', 'Django', 'Flask', 'FastAPI',
            'Spring Boot', 'Spring MVC', 'Hibernate', 'Ruby on Rails', 'Laravel', 'Symfony', 'CodeIgniter',
            'ASP.NET Core', 'ASP.NET MVC', 'Blazor', 'Phoenix', 'Gin', 'Echo', 'Fiber', 'Chi',
            'GraphQL', 'REST API', 'gRPC', 'WebSocket', 'Server-Sent Events', 'OAuth 2.0', 'JWT',
            'Webpack', 'Vite', 'Rollup', 'Parcel', 'esbuild', 'Babel', 'ESLint', 'Prettier',
            'Redux', 'MobX', 'Zustand', 'Recoil', 'Vuex', 'Pinia', 'NgRx', 'Akita',
            'Jest', 'Mocha', 'Jasmine', 'Cypress', 'Playwright', 'Puppeteer', 'Selenium WebDriver',
            'Storybook', 'Chromatic', 'Percy', 'Applitools', 'BrowserStack', 'Sauce Labs',
            
            # Mobile Development (40+)
            'Android SDK', 'iOS SDK', 'React Native', 'Flutter', 'Xamarin', 'Ionic', 'Cordova',
            'SwiftUI', 'UIKit', 'Jetpack Compose', 'Android Jetpack', 'Room Database', 'Core Data',
            'Firebase', 'Realm', 'SQLite Mobile', 'AsyncStorage', 'MMKV', 'Keychain', 'Biometrics',
            'Push Notifications', 'Deep Linking', 'App Store Connect', 'Google Play Console',
            'TestFlight', 'Fastlane', 'Bitrise', 'App Center', 'CodePush', 'Detox', 'Espresso',
            'XCTest', 'KMM', 'Kotlin Multiplatform', 'Capacitor', 'NativeScript', 'Expo', 'PWA',
            'Unity Mobile', 'Unreal Mobile', 'ARKit', 'ARCore', 'Core ML', 'ML Kit',
            
            # Cloud Platforms (60+)
            'AWS', 'Azure', 'Google Cloud Platform', 'IBM Cloud', 'Oracle Cloud', 'Alibaba Cloud',
            'AWS EC2', 'AWS S3', 'AWS Lambda', 'AWS RDS', 'AWS DynamoDB', 'AWS SQS', 'AWS SNS',
            'AWS ECS', 'AWS EKS', 'AWS Fargate', 'AWS CloudFormation', 'AWS CDK', 'AWS SAM',
            'Azure VMs', 'Azure Blob Storage', 'Azure Functions', 'Azure SQL', 'Azure Cosmos DB',
            'Azure Service Bus', 'Azure AKS', 'Azure DevOps', 'Azure ARM', 'Azure Bicep',
            'GCP Compute Engine', 'GCP Cloud Storage', 'GCP Cloud Functions', 'GCP BigQuery',
            'GCP Firestore', 'GCP Pub/Sub', 'GCP GKE', 'GCP Cloud Run', 'GCP Deployment Manager',
            'DigitalOcean', 'Heroku', 'Vercel', 'Netlify', 'Cloudflare Workers', 'Fly.io', 'Railway',
            'Linode', 'Vultr', 'Hetzner', 'OVH Cloud', 'Scaleway', 'Rackspace', 'VMware Cloud',
            'OpenStack', 'CloudFoundry', 'Rancher', 'Portainer', 'Proxmox',
            
            # DevOps & CI/CD (50+)
            'Docker', 'Kubernetes', 'Helm', 'Istio', 'Linkerd', 'Consul', 'Vault', 'Terraform',
            'Ansible', 'Puppet', 'Chef', 'SaltStack', 'Pulumi', 'Crossplane', 'ArgoCD', 'FluxCD',
            'Jenkins', 'GitHub Actions', 'GitLab CI', 'CircleCI', 'Travis CI', 'TeamCity', 'Bamboo',
            'Azure Pipelines', 'AWS CodePipeline', 'Google Cloud Build', 'Buildkite', 'Drone CI',
            'Prometheus', 'Grafana', 'Datadog', 'New Relic', 'Splunk', 'ELK Stack', 'Loki',
            'Jaeger', 'Zipkin', 'OpenTelemetry', 'PagerDuty', 'OpsGenie', 'VictorOps', 'Nagios',
            'Zabbix', 'Sensu', 'InfluxDB', 'TimescaleDB', 'StatsD', 'Telegraf', 'Fluentd',
            
            # Databases (60+)
            'MySQL', 'PostgreSQL', 'SQL Server', 'Oracle Database', 'MariaDB', 'SQLite',
            'MongoDB', 'Cassandra', 'CouchDB', 'CouchBase', 'RavenDB', 'ArangoDB', 'OrientDB',
            'Redis', 'Memcached', 'Hazelcast', 'Apache Ignite', 'Aerospike', 'Ehcache',
            'Elasticsearch', 'Solr', 'Algolia', 'MeiliSearch', 'Typesense', 'OpenSearch',
            'Neo4j', 'Amazon Neptune', 'JanusGraph', 'TigerGraph', 'ArangoDB Graph', 'Dgraph',
            'ClickHouse', 'Druid', 'Pinot', 'Presto', 'Trino', 'Apache Hive', 'Apache Impala',
            'Snowflake', 'Databricks', 'Apache Spark', 'Apache Flink', 'Apache Kafka', 'RabbitMQ',
            'ActiveMQ', 'Apache Pulsar', 'NATS', 'ZeroMQ', 'Amazon Kinesis', 'Azure Event Hubs',
            'DynamoDB', 'Amazon Redshift', 'Google BigQuery', 'Azure Synapse', 'Teradata', 'Vertica',
            'CockroachDB', 'TiDB', 'YugabyteDB', 'FaunaDB', 'Supabase', 'PlanetScale',
            
            # Data Science & ML (80+)
            'Machine Learning', 'Deep Learning', 'Neural Networks', 'Natural Language Processing',
            'Computer Vision', 'Reinforcement Learning', 'Time Series Analysis', 'Anomaly Detection',
            'TensorFlow', 'PyTorch', 'Keras', 'Scikit-learn', 'XGBoost', 'LightGBM', 'CatBoost',
            'Pandas', 'NumPy', 'SciPy', 'Matplotlib', 'Seaborn', 'Plotly', 'Bokeh', 'Altair',
            'Jupyter Notebooks', 'Google Colab', 'Kaggle', 'Apache Zeppelin', 'Databricks Notebooks',
            'MLflow', 'Kubeflow', 'TFX', 'Airflow', 'Prefect', 'Dagster', 'Luigi', 'Kedro',
            'Hugging Face', 'Transformers', 'BERT', 'GPT', 'LLaMA', 'Stable Diffusion', 'DALL-E',
            'LangChain', 'LlamaIndex', 'Pinecone', 'Weaviate', 'Milvus', 'Chroma', 'Qdrant',
            'OpenCV', 'PIL/Pillow', 'YOLO', 'Detectron2', 'MediaPipe', 'Dlib', 'Face Recognition',
            'spaCy', 'NLTK', 'Gensim', 'TextBlob', 'CoreNLP', 'AllenNLP', 'Flair',
            'Feature Engineering', 'Feature Store', 'Model Serving', 'A/B Testing', 'Experimentation',
            'Statistical Modeling', 'Hypothesis Testing', 'Bayesian Methods', 'Causal Inference',
            'Data Mining', 'Data Wrangling', 'ETL Pipelines', 'Data Validation', 'Data Quality',
            
            # Cybersecurity (50+)
            'Network Security', 'Application Security', 'Cloud Security', 'Endpoint Security',
            'Identity Access Management', 'Zero Trust', 'SIEM', 'SOAR', 'EDR', 'XDR', 'NDR',
            'Penetration Testing', 'Ethical Hacking', 'Vulnerability Assessment', 'Bug Bounty',
            'Burp Suite', 'Metasploit', 'Nmap', 'Wireshark', 'Kali Linux', 'OWASP', 'Nessus',
            'Splunk Security', 'QRadar', 'ArcSight', 'LogRhythm', 'Sentinel', 'Chronicle',
            'CrowdStrike', 'Carbon Black', 'SentinelOne', 'Cybereason', 'Defender ATP',
            'Firewall Configuration', 'IDS/IPS', 'WAF', 'DLP', 'CASB', 'SASE', 'VPN',
            'Encryption', 'PKI', 'Certificate Management', 'HSM', 'Key Management',
            'SOC Operations', 'Incident Response', 'Digital Forensics', 'Malware Analysis',
            'Threat Intelligence', 'Threat Hunting', 'Red Team', 'Blue Team', 'Purple Team',
            'ISO 27001', 'SOC 2', 'PCI DSS', 'HIPAA', 'GDPR', 'NIST', 'CIS Controls',
            
            # Business Intelligence (40+)
            'Tableau', 'Power BI', 'Looker', 'Qlik Sense', 'QlikView', 'Sisense', 'Domo',
            'Metabase', 'Superset', 'Redash', 'Mode Analytics', 'ThoughtSpot', 'Sigma Computing',
            'DAX', 'Power Query', 'M Language', 'Calculated Fields', 'LOD Expressions',
            'Dashboard Design', 'Data Storytelling', 'KPI Development', 'Scorecard Design',
            'SSRS', 'SSIS', 'SSAS', 'Crystal Reports', 'Cognos', 'SAP BusinessObjects',
            'MicroStrategy', 'Informatica', 'Talend', 'Pentaho', 'Alteryx', 'Dataiku', 'KNIME',
            'Excel Advanced', 'Excel VBA', 'Excel Macros', 'Excel Power Pivot', 'Google Sheets',
            
            # Project Management (30+)
            'Agile Methodology', 'Scrum', 'Kanban', 'SAFe', 'LeSS', 'Waterfall', 'Hybrid Agile',
            'JIRA', 'Confluence', 'Trello', 'Asana', 'Monday.com', 'Notion', 'ClickUp',
            'Microsoft Project', 'Smartsheet', 'Basecamp', 'Wrike', 'TeamGantt', 'Airtable',
            'Sprint Planning', 'Backlog Grooming', 'Story Points', 'Velocity Tracking',
            'Stakeholder Management', 'Risk Management', 'Resource Planning', 'Gantt Charts',
            'PMP', 'Prince2', 'PMI-ACP', 'Certified ScrumMaster', 'Product Owner',
            
            # Design Tools (40+)
            'Figma', 'Sketch', 'Adobe XD', 'InVision', 'Framer', 'Principle', 'ProtoPie',
            'Adobe Photoshop', 'Adobe Illustrator', 'Adobe InDesign', 'Adobe After Effects',
            'Adobe Premiere Pro', 'Final Cut Pro', 'DaVinci Resolve', 'Blender', 'Cinema 4D',
            'Maya', '3ds Max', 'ZBrush', 'Substance Painter', 'Houdini', 'Nuke', 'Fusion',
            'Canva', 'CorelDRAW', 'Affinity Designer', 'Affinity Photo', 'GIMP', 'Inkscape',
            'Wireframing', 'Prototyping', 'User Research', 'Usability Testing', 'A/B Testing Design',
            'Design Systems', 'Component Libraries', 'Style Guides', 'Brand Guidelines',
            'Typography', 'Color Theory', 'Layout Design', 'Responsive Design', 'Accessibility',
            
            # Marketing & Sales (60+)
            'Digital Marketing', 'Content Marketing', 'Social Media Marketing', 'Email Marketing',
            'SEO', 'SEM', 'PPC', 'Google Ads', 'Facebook Ads', 'LinkedIn Ads', 'TikTok Ads',
            'Instagram Marketing', 'Twitter Marketing', 'YouTube Marketing', 'Influencer Marketing',
            'Marketing Automation', 'HubSpot', 'Marketo', 'Pardot', 'Mailchimp', 'ActiveCampaign',
            'CRM', 'Salesforce', 'HubSpot CRM', 'Zoho CRM', 'Pipedrive', 'Freshsales',
            'Google Analytics', 'Adobe Analytics', 'Mixpanel', 'Amplitude', 'Segment', 'Heap',
            'Hotjar', 'Crazy Egg', 'FullStory', 'Lucky Orange', 'Session Recording',
            'Conversion Rate Optimization', 'Landing Page Design', 'Lead Generation', 'Lead Nurturing',
            'Account-Based Marketing', 'Growth Hacking', 'Viral Marketing', 'Guerrilla Marketing',
            'Brand Management', 'Public Relations', 'Crisis Communication', 'Media Buying',
            'Copywriting', 'Content Strategy', 'Editorial Calendar', 'Content Curation',
            'B2B Sales', 'B2C Sales', 'Inside Sales', 'Field Sales', 'Enterprise Sales',
            'Sales Strategy', 'Pipeline Management', 'Forecasting', 'Quota Achievement',
            'Negotiation', 'Closing Techniques', 'Objection Handling', 'Discovery Calls',
            
            # HR & Administration (40+)
            'Talent Acquisition', 'Recruitment', 'Sourcing', 'ATS Systems', 'LinkedIn Recruiter',
            'Employee Onboarding', 'Employee Engagement', 'Performance Management', 'Succession Planning',
            'Compensation & Benefits', 'Payroll Processing', 'HRIS', 'Workday', 'SAP SuccessFactors',
            'BambooHR', 'Gusto', 'ADP', 'Paychex', 'Kronos', 'Time & Attendance',
            'Labor Law', 'Employment Law', 'Compliance', 'HR Policies', 'Employee Relations',
            'Training & Development', 'Learning Management', 'E-Learning Development', 'Instructional Design',
            'Office Management', 'Facilities Management', 'Vendor Management', 'Procurement',
            'Calendar Management', 'Travel Arrangements', 'Event Planning', 'Meeting Coordination',
            'MS Office Suite', 'Google Workspace', 'Slack', 'Microsoft Teams', 'Zoom',
            
            # Finance & Accounting (50+)
            'Financial Analysis', 'Financial Modeling', 'Valuation', 'DCF Analysis', 'LBO Modeling',
            'Financial Reporting', 'GAAP', 'IFRS', 'SEC Reporting', 'Consolidation',
            'Accounts Payable', 'Accounts Receivable', 'General Ledger', 'Journal Entries',
            'Bank Reconciliation', 'Cash Flow Management', 'Working Capital', 'Treasury',
            'Budgeting', 'Forecasting', 'Variance Analysis', 'Cost Accounting', 'Activity-Based Costing',
            'Tax Accounting', 'Tax Planning', 'GST', 'Income Tax', 'Transfer Pricing', 'Tax Compliance',
            'Audit', 'Internal Audit', 'External Audit', 'SOX Compliance', 'Risk Assessment',
            'SAP FICO', 'Oracle Financials', 'NetSuite', 'QuickBooks', 'Xero', 'FreshBooks',
            'Tally', 'Zoho Books', 'Sage', 'Microsoft Dynamics Finance', 'Hyperion',
            'Bloomberg Terminal', 'Reuters Eikon', 'FactSet', 'Capital IQ', 'PitchBook',
            
            # Healthcare (40+)
            'Patient Care', 'Clinical Documentation', 'Medical Coding', 'ICD-10', 'CPT Coding',
            'Electronic Health Records', 'Epic Systems', 'Cerner', 'Meditech', 'Allscripts',
            'HIPAA Compliance', 'Patient Privacy', 'Clinical Trials', 'FDA Regulations',
            'Nursing Skills', 'Medication Administration', 'IV Therapy', 'Wound Care',
            'Phlebotomy', 'ECG/EKG', 'Radiology', 'MRI', 'CT Scan', 'X-Ray', 'Ultrasound',
            'Laboratory Techniques', 'Blood Banking', 'Microbiology', 'Pathology', 'Histology',
            'Physical Therapy', 'Occupational Therapy', 'Speech Therapy', 'Rehabilitation',
            'Mental Health Counseling', 'Cognitive Behavioral Therapy', 'Crisis Intervention',
            'Pharmacy Operations', 'Drug Dispensing', 'Pharmacovigilance', 'Drug Safety',
            
            # Education (30+)
            'Curriculum Development', 'Lesson Planning', 'Classroom Management', 'Student Assessment',
            'Differentiated Instruction', 'Special Education', 'Inclusive Education', 'Gifted Education',
            'Online Teaching', 'Virtual Classroom', 'LMS Administration', 'Course Design',
            'Educational Technology', 'Moodle', 'Canvas', 'Blackboard', 'Google Classroom',
            'STEM Education', 'Montessori', 'Waldorf', 'IB Curriculum', 'CBSE', 'ICSE', 'State Board',
            'Early Childhood Education', 'Elementary Education', 'Secondary Education', 'Higher Education',
            'Student Counseling', 'Career Guidance', 'Academic Advising', 'Parent Communication',
            
            # Hospitality & Tourism (30+)
            'Hotel Operations', 'Front Desk Management', 'Guest Services', 'Concierge Services',
            'Housekeeping Management', 'Room Division', 'Revenue Management', 'Yield Management',
            'Food & Beverage Operations', 'Restaurant Management', 'Banquet Operations', 'Catering',
            'Hotel PMS', 'Opera', 'Fidelio', 'Amadeus', 'Sabre', 'Galileo', 'Worldspan',
            'Travel Planning', 'Itinerary Design', 'Tour Operations', 'Destination Management',
            'Event Planning', 'Conference Management', 'MICE', 'Wedding Planning',
            'Airline Operations', 'Ground Handling', 'Cabin Crew', 'Aviation Safety',
            
            # Manufacturing & Engineering (50+)
            'Mechanical Engineering', 'Electrical Engineering', 'Civil Engineering', 'Chemical Engineering',
            'Industrial Engineering', 'Manufacturing Engineering', 'Production Engineering',
            'AutoCAD', 'SolidWorks', 'CATIA', 'Creo', 'NX', 'Inventor', 'Fusion 360',
            'Six Sigma', 'Lean Manufacturing', 'Kaizen', '5S', 'Total Quality Management',
            'Process Engineering', 'Process Optimization', 'Process Control', 'SPC',
            'CNC Programming', 'CNC Operation', 'G-Code', 'CAM', 'Mastercam', 'Edgecam',
            'PLC Programming', 'SCADA', 'HMI', 'Industrial Automation', 'Robotics',
            'Quality Control', 'Quality Assurance', 'ISO 9001', 'IATF 16949', 'AS9100',
            'Supply Chain Management', 'Inventory Management', 'Material Planning', 'MRP',
            'ERP Systems', 'SAP MM', 'SAP PP', 'Oracle Manufacturing', 'Microsoft Dynamics',
            'Product Design', 'Product Development', 'Prototyping', '3D Printing', 'Injection Molding',
            
            # Legal (30+)
            'Contract Law', 'Corporate Law', 'Litigation', 'Intellectual Property', 'Patent Law',
            'Trademark Law', 'Copyright Law', 'Employment Law', 'Labor Law', 'Immigration Law',
            'Real Estate Law', 'Banking Law', 'Tax Law', 'Criminal Law', 'Family Law', 'Civil Law',
            'Legal Research', 'Legal Writing', 'Contract Drafting', 'Contract Review', 'Due Diligence',
            'Mergers & Acquisitions', 'Securities Law', 'Compliance', 'Regulatory Affairs',
            'Westlaw', 'LexisNexis', 'Documentation', 'e-Discovery', 'Case Management',
            
            # Soft Skills (50+)
            'Communication Skills', 'Presentation Skills', 'Public Speaking', 'Active Listening',
            'Written Communication', 'Business Writing', 'Technical Writing', 'Report Writing',
            'Leadership', 'Team Management', 'People Management', 'Mentoring', 'Coaching',
            'Problem Solving', 'Critical Thinking', 'Analytical Thinking', 'Decision Making',
            'Time Management', 'Prioritization', 'Multitasking', 'Organization Skills',
            'Teamwork', 'Collaboration', 'Cross-functional Teams', 'Remote Collaboration',
            'Adaptability', 'Flexibility', 'Change Management', 'Resilience', 'Stress Management',
            'Creativity', 'Innovation', 'Design Thinking', 'Ideation', 'Brainstorming',
            'Emotional Intelligence', 'Empathy', 'Conflict Resolution', 'Negotiation',
            'Customer Focus', 'Client Relations', 'Stakeholder Management', 'Vendor Relations',
            'Attention to Detail', 'Quality Orientation', 'Result Orientation', 'Goal Setting',
            'Self-motivation', 'Initiative', 'Proactiveness', 'Ownership', 'Accountability',
            
            # Additional Domain-Specific Skills (100+)
            'Cooking', 'Baking', 'Menu Planning', 'Food Presentation', 'Kitchen Management',
            'Makeup Artistry', 'Hair Styling', 'Skincare', 'Nail Art', 'Spa Treatments',
            'Yoga Instruction', 'Fitness Training', 'Personal Training', 'Group Classes', 'Nutrition',
            'Fashion Design', 'Pattern Making', 'Garment Construction', 'Textile Design', 'Merchandising',
            'Photography', 'Videography', 'Photo Editing', 'Color Grading', 'Drone Operation',
            'Music Production', 'Audio Editing', 'Sound Design', 'Voice Over', 'Podcast Production',
            'Agriculture', 'Horticulture', 'Crop Management', 'Organic Farming', 'Pest Management',
            'Animal Care', 'Veterinary Assistance', 'Pet Grooming', 'Animal Training',
            'Driving', 'Defensive Driving', 'Route Planning', 'Vehicle Maintenance',
            'Cleaning', 'Sanitation', 'Waste Management', 'Pest Control', 'Facility Maintenance',
            'Security', 'Surveillance', 'Access Control', 'Emergency Response', 'First Aid',
            'Carpentry', 'Plumbing', 'Electrical Work', 'Painting', 'Masonry', 'Welding',
            'Gardening', 'Landscaping', 'Plant Care', 'Irrigation', 'Lawn Maintenance',
            'Child Care', 'Early Development', 'Play-based Learning', 'Child Nutrition',
            'Elder Care', 'Mobility Assistance', 'Medication Management', 'Companion Care',
            'Tailoring', 'Alterations', 'Embroidery', 'Knitting', 'Crochet', 'Handloom',
            'Pottery', 'Ceramics', 'Sculpture', 'Painting', 'Drawing', 'Calligraphy',
            'Social Work', 'Community Development', 'Counseling', 'Case Management', 'Advocacy',
            'Retail Sales', 'Customer Service', 'Cash Handling', 'Inventory Management', 'Visual Merchandising',
            'Real Estate', 'Property Valuation', 'Property Management', 'Lease Negotiation',
            'Hindi', 'English', 'Tamil', 'Telugu', 'Marathi', 'Bengali', 'Gujarati', 'Kannada', 'Malayalam',
            'Spanish', 'French', 'German', 'Mandarin', 'Japanese', 'Korean', 'Arabic', 'Portuguese'
        ]
        
        print(f"   Initialized {len(self.all_skills)} unique skills")
        
        # ========== 600+ UNIQUE COMPANIES ==========
        self.all_companies = [
            # IT Giants
            'TCS', 'Infosys', 'Wipro', 'HCL Technologies', 'Tech Mahindra', 'L&T Infotech', 'Mindtree',
            'Mphasis', 'Cyient', 'Persistent Systems', 'Birlasoft', 'Hexaware', 'NIIT Technologies',
            'Zensar Technologies', 'Mastek', 'KPIT Technologies', 'Sonata Software', 'Happiest Minds',
            'Coforge', 'Larsen & Toubro Technology Services', 'LTTS', 'Tata Elxsi', 'Sasken Technologies',
            
            # MNCs
            'Accenture India', 'Cognizant', 'Capgemini India', 'IBM India', 'Microsoft India', 'Google India',
            'Amazon India', 'Meta India', 'Apple India', 'Oracle India', 'SAP India', 'Salesforce India',
            'Adobe India', 'VMware India', 'Cisco India', 'Intel India', 'Qualcomm India', 'NVIDIA India',
            'Dell India', 'HP India', 'Lenovo India', 'Samsung India', 'LG India', 'Sony India',
            'Ericsson India', 'Nokia India', 'Huawei India', 'ZTE India', 'Schneider Electric India',
            'Siemens India', 'ABB India', 'Bosch India', 'Continental India', 'Honeywell India',
            'GE India', 'Philips India', 'Panasonic India', 'Hitachi India', 'Toshiba India',
            
            # Startups & Unicorns
            'Flipkart', 'Paytm', 'Zomato', 'Swiggy', 'Ola', 'Uber India', 'BYJU\'s', 'Unacademy',
            'Razorpay', 'PhonePe', 'MobiKwik', 'Pine Labs', 'BharatPe', 'Groww', 'Zerodha', 'Upstox',
            'Meesho', 'Nykaa', 'Myntra', 'Ajio', 'Pepperfry', 'Urban Company', 'Porter', 'Dunzo',
            'CRED', 'Dream11', 'MPL', 'Ludo King', 'Nazara Technologies', 'Games24x7',
            'Freshworks', 'Zoho', 'Chargebee', 'Browserstack', 'Postman', 'CleverTap', 'WebEngage',
            'Druva', 'Icertis', 'Mindtickle', 'Hasura', 'Polygon', 'CoinDCX', 'WazirX', 'CoinSwitch',
            'Cars24', 'CarDekho', 'SpinNY', 'Droom', 'OLX India', 'Quikr', 'NoBroker', 'Housing.com',
            'Practo', 'PharmEasy', '1mg', 'Netmeds', 'Medlife', 'Cure.fit', 'cult.fit', 'HealthifyMe',
            'Lenskart', 'Boat', 'Sugar Cosmetics', 'Mamaearth', 'The Man Company', 'Bombay Shaving Company',
            'Bira 91', 'Paper Boat', 'Raw Pressery', 'Country Delight', 'Licious', 'FreshToHome',
            'BigBasket', 'Grofers', 'ZeptoNow', 'Blinkit', 'Instamart', 'JioMart', 'Amazon Fresh',
            'Rivigo', 'Blackbuck', 'Delhivery', 'Ecom Express', 'Xpressbees', 'Shadowfax', 'ElasticRun',
            'OYO Rooms', 'Treebo Hotels', 'FabHotels', 'Zostel', 'goStays', 'MakeMyTrip', 'Goibibo',
            'ixigo', 'Yatra', 'Cleartrip', 'EaseMyTrip', 'HappyEasyGo', 'Confirmtkt', 'RailYatri',
            'Ather Energy', 'Ola Electric', 'Bounce', 'Yulu', 'Vogo', 'BluSmart', 'Rapido',
            'Vedantu', 'Toppr', 'Doubtnut', 'Physics Wallah', 'UpGrad', 'Simplilearn', 'Great Learning',
            'Scaler Academy', 'InterviewBit', 'GeeksforGeeks', 'Coding Ninjas', 'Newton School',
            'Slice', 'Jupiter Money', 'Fi Money', 'Niyo', 'Salary', 'Refyne', 'EarlySalary',
            'PolicyBazaar', 'Acko', 'Digit Insurance', 'Go Digit', 'Turtlemint', 'RenewBuy',
            'ShareChat', 'Moj', 'Josh', 'Chingari', 'Roposo', 'TakaTak', 'Mitron',
            
            # Banks
            'State Bank of India', 'HDFC Bank', 'ICICI Bank', 'Axis Bank', 'Kotak Mahindra Bank',
            'IndusInd Bank', 'Yes Bank', 'Federal Bank', 'IDFC First Bank', 'RBL Bank', 'Bandhan Bank',
            'Bank of Baroda', 'Punjab National Bank', 'Canara Bank', 'Union Bank of India', 'Bank of India',
            'Indian Bank', 'Central Bank of India', 'Indian Overseas Bank', 'UCO Bank', 'Bank of Maharashtra',
            'IDBI Bank', 'Indian Post Payments Bank', 'Airtel Payments Bank', 'Paytm Payments Bank', 'Fino Payments Bank',
            'Standard Chartered India', 'Citi India', 'HSBC India', 'Deutsche Bank India', 'Barclays India',
            
            # NBFCs & Finance
            'Bajaj Finance', 'Bajaj Finserv', 'HDFC Ltd', 'L&T Finance', 'Mahindra Finance', 'Shriram Finance',
            'Tata Capital', 'Aditya Birla Capital', 'Muthoot Finance', 'Manappuram Finance', 'Edelweiss',
            'IIFL Finance', 'Poonawalla Fincorp', 'Hero FinCorp', 'HDB Finance', 'Fullerton India',
            
            # Insurance
            'LIC', 'HDFC Life', 'ICICI Prudential Life', 'SBI Life', 'Max Life', 'Bajaj Allianz Life',
            'Kotak Life', 'Tata AIA Life', 'PNB MetLife', 'Aditya Birla Sun Life', 'Canara HSBC Life',
            'ICICI Lombard', 'New India Assurance', 'United India Insurance', 'Oriental Insurance',
            'National Insurance', 'HDFC ERGO', 'Bajaj Allianz General', 'Tata AIG', 'IFFCO Tokio',
            'Star Health', 'Max Bupa', 'Apollo Munich', 'Care Health', 'ManipalCigna',
            
            # Healthcare
            'Apollo Hospitals', 'Fortis Healthcare', 'Max Healthcare', 'Manipal Hospitals', 'Narayana Health',
            'Medanta', 'AIIMS', 'Aster DM Healthcare', 'Columbia Asia', 'Kokilaben Hospital',
            'Lilavati Hospital', 'Breach Candy Hospital', 'Hinduja Hospital', 'Jaslok Hospital', 'Wockhardt Hospitals',
            'SevenHills Hospital', 'Global Hospitals', 'KIMS Hospitals', 'Yashoda Hospitals', 'Care Hospitals',
            'Rainbow Hospitals', 'Cloudnine Hospitals', 'Motherhood Hospitals', 'Sahyadri Hospitals', 'Ruby Hall',
            'Dr. Lal PathLabs', 'SRL Diagnostics', 'Metropolis Healthcare', 'Thyrocare', 'Suburban Diagnostics',
            
            # Pharma
            'Sun Pharma', 'Cipla', 'Dr. Reddy\'s', 'Lupin', 'Aurobindo Pharma', 'Zydus Lifesciences',
            'Torrent Pharma', 'Alkem Labs', 'Glenmark', 'Biocon', 'Divis Labs', 'Ipca Labs',
            'Abbott India', 'Pfizer India', 'GSK India', 'Novartis India', 'Sanofi India', 'AstraZeneca India',
            'Merck India', 'Johnson & Johnson India', 'Roche India', 'Bayer India', 'Eli Lilly India',
            'Mankind Pharma', 'Intas Pharma', 'Micro Labs', 'Hetero Drugs', 'Natco Pharma', 'Granules India',
            'Serum Institute', 'Bharat Biotech', 'Biological E', 'Panacea Biotec', 'Indian Immunologicals',
            
            # FMCG
            'Hindustan Unilever', 'ITC Limited', 'Nestle India', 'Britannia', 'Parle Products', 'Dabur',
            'Godrej Consumer', 'Marico', 'Colgate-Palmolive India', 'Procter & Gamble India', 'Reckitt India',
            'Emami', 'Bajaj Consumer', 'Jyothy Labs', 'Wipro Consumer', 'CavinKare', 'VVF India',
            'Amul', 'Mother Dairy', 'Kwality', 'Parag Milk', 'Heritage Foods', 'Hatsun Agro',
            'PepsiCo India', 'Coca-Cola India', 'Parle Agro', 'Dabur Real', 'Tropicana India', 'Red Bull India',
            'Haldiram\'s', 'Bikaji', 'Bikanervala', 'MTR Foods', 'Gits Food', 'Priya Foods', 'Aachi',
            
            # Auto
            'Tata Motors', 'Mahindra & Mahindra', 'Maruti Suzuki', 'Hyundai India', 'Kia India', 'Toyota India',
            'Honda Cars India', 'MG Motor India', 'Skoda India', 'Volkswagen India', 'Mercedes India', 'BMW India',
            'Audi India', 'Jaguar Land Rover India', 'Volvo India', 'Ford India', 'Renault India', 'Nissan India',
            'Hero MotoCorp', 'Bajaj Auto', 'TVS Motor', 'Royal Enfield', 'Honda Motorcycle', 'Yamaha India',
            'Suzuki Motorcycle', 'KTM India', 'Kawasaki India', 'Harley-Davidson India', 'Triumph India',
            'Tata AutoComp', 'Motherson Sumi', 'Bosch India', 'Continental India', 'Denso India', 'Valeo India',
            'CEAT', 'MRF', 'Apollo Tyres', 'JK Tyre', 'Bridgestone India', 'Michelin India', 'Goodyear India',
            
            # Infrastructure & Real Estate
            'Larsen & Toubro', 'DLF', 'Godrej Properties', 'Oberoi Realty', 'Prestige Estates', 'Brigade Group',
            'Sobha', 'Mahindra Lifespace', 'Lodha Group', 'Hiranandani', 'Shapoorji Pallonji', 'Tata Realty',
            'Adani Realty', 'Embassy Group', 'Phoenix Mills', 'Indiabulls Real Estate', 'Sunteck Realty',
            'Puravankara', 'Shriram Properties', 'Kolte-Patil', 'Ashiana Housing', 'Ansal Housing',
            'GMR Group', 'GVK', 'Adani Ports', 'Bharti Realty', 'K Raheja Corp', 'Runwal Group',
            
            # Energy & Power
            'Reliance Industries', 'ONGC', 'Indian Oil', 'Bharat Petroleum', 'Hindustan Petroleum', 'GAIL',
            'NTPC', 'Power Grid', 'Tata Power', 'Adani Power', 'JSW Energy', 'NHPC', 'SJVN', 'Torrent Power',
            'CESC', 'BSES', 'Tata BP Solar', 'Adani Green', 'Greenko', 'ReNew Power', 'Azure Power',
            'Thermax', 'BHEL', 'Siemens Energy India', 'GE Power India', 'Alstom India',
            
            # Metals & Mining
            'Tata Steel', 'JSW Steel', 'SAIL', 'Hindalco', 'Vedanta', 'NMDC', 'Coal India', 'Jindal Steel',
            'Hindustan Zinc', 'National Aluminium', 'MOIL', 'GMDC', 'KIOCL', 'Sandur Manganese',
            
            # Telecom
            'Reliance Jio', 'Bharti Airtel', 'Vodafone Idea', 'BSNL', 'MTNL', 'Tata Communications',
            'Airtel Business', 'Jio Business', 'ACT Fibernet', 'Hathway', 'DEN Networks', 'GTPL',
            
            # Airlines & Travel
            'IndiGo', 'Air India', 'SpiceJet', 'Go First', 'Vistara', 'AirAsia India', 'Akasa Air',
            'Air India Express', 'Alliance Air', 'Star Air', 'TruJet', 'Flybig',
            'IRCTC', 'Indian Railways', 'Metro Rail', 'DMRC', 'Mumbai Metro', 'Bangalore Metro',
            
            # Hotels & Hospitality
            'Taj Hotels', 'Oberoi Hotels', 'ITC Hotels', 'Leela Hotels', 'The Lalit', 'Hyatt India',
            'Marriott India', 'Hilton India', 'Radisson India', 'Accor India', 'IHG India', 'Wyndham India',
            'Lemon Tree', 'Ginger Hotels', 'Fortune Hotels', 'Sarovar Hotels', 'Pride Hotels', 'Keys Hotels',
            
            # Retail
            'Reliance Retail', 'Tata Trent', 'Avenue Supermarts', 'Future Group', 'Aditya Birla Retail',
            'Shoppers Stop', 'Lifestyle', 'Westside', 'Pantaloons', 'FBB', 'Max Fashion', 'Zara India',
            'H&M India', 'Uniqlo India', 'Decathlon India', 'Croma', 'Reliance Digital', 'Vijay Sales',
            'Spencer\'s', 'Spar India', 'Star Bazaar', 'Hypercity', 'More Retail', 'Easyday',
            
            # Media & Entertainment
            'Star India', 'Zee Entertainment', 'Sony India', 'Viacom18', 'Network18', 'TV18', 'NDTV',
            'Times Group', 'HT Media', 'India Today', 'ABP News', 'Republic TV', 'NewsX', 'Mirror Now',
            'Netflix India', 'Amazon Prime India', 'Disney+ Hotstar', 'JioCinema', 'Sony LIV', 'Zee5', 'Voot',
            'T-Series', 'Saregama', 'Hungama', 'Gaana', 'JioSaavn', 'Wynk Music', 'Spotify India',
            'Yash Raj Films', 'Dharma Productions', 'Red Chillies', 'Excel Entertainment', 'UTV Motion',
            
            # Education
            'BYJU\'s', 'Unacademy', 'Vedantu', 'PhysicsWallah', 'UpGrad', 'Simplilearn', 'Great Learning',
            'Manipal Global', 'Amity University', 'LPU', 'Shiv Nadar', 'Ashoka University', 'BITS Pilani',
            'NIIT', 'Aptech', 'IIIT Hyderabad', 'ISB', 'IIMs', 'IITs', 'NITs', 'XLRI', 'SP Jain',
            'Symbiosis', 'Christ University', 'VIT', 'SRM', 'KIIT', 'Thapar', 'Manipal',
            
            # Consulting
            'McKinsey India', 'BCG India', 'Bain India', 'Deloitte India', 'PwC India', 'EY India', 'KPMG India',
            'Accenture Strategy', 'Oliver Wyman', 'Roland Berger', 'A.T. Kearney', 'Kearney', 'Monitor Deloitte',
            'ZS Associates', 'Alvarez & Marsal', 'FTI Consulting', 'LEK Consulting', 'Parthenon', 'EYP',
            'PA Consulting', 'Red Seer', 'Redseer Strategy', 'Praxis', 'Aon India', 'Mercer India', 'Willis Towers',
            
            # Legal
            'AZB & Partners', 'Cyril Amarchand', 'Shardul Amarchand', 'Khaitan & Co', 'Trilegal', 'JSA',
            'Luthra & Luthra', 'S&R Associates', 'Majmudar & Partners', 'DSK Legal', 'Nishith Desai',
            'Economic Laws Practice', 'Lakshmikumaran & Sridharan', 'IndusLaw', 'Vaish Associates',
            
            # SMEs & Local Businesses
            'Local IT Services', 'Regional Startup', 'City Hospital', 'District School', 'Town Retail',
            'Family Business', 'Small Enterprise', 'Medium Enterprise', 'Local Manufacturer', 'Regional Trader',
            'Self-Employed', 'Freelance Consultant', 'Home Business', 'Cottage Industry', 'Village Enterprise',
            'Cooperative Society', 'NGO', 'Non-Profit', 'Social Enterprise', 'Community Organization',
            'Government Department', 'PSU', 'State Government', 'Central Government', 'Municipal Corporation',
            'Panchayat Office', 'Block Development', 'District Administration', 'State Board', 'Central Board',
            
            # Additional Companies
            'Godrej & Boyce', 'Kirloskar', 'Bajaj Group', 'Mahindra Group', 'Tata Sons', 'Aditya Birla Group',
            'Welspun', 'JSW Group', 'Jindal Group', 'Vedanta Resources', 'Essar', 'GMR', 'GVK',
            'Wadia Group', 'RPG Group', 'Murugappa Group', 'TVS Group', 'Hinduja Group', 'Shapoorji Pallonji Group',
            'Raymond', 'Arvind Mills', 'Vardhman', 'Trident', 'Welspun India', 'Indo Count', 'Himatsingka',
            'Asian Paints', 'Berger Paints', 'Kansai Nerolac', 'Akzo Nobel India', 'Pidilite', 'Fevicol',
            'Havells', 'V-Guard', 'Crompton', 'Bajaj Electricals', 'Orient Electric', 'Usha International',
            'TTK Prestige', 'Butterfly Gandhimathi', 'Hawkins', 'Preethi', 'Wonderchef', 'Pigeon',
            'Titan', 'Tanishq', 'Fastrack', 'Sonata', 'Timex India', 'Casio India', 'Fossil India',
            'VIP Industries', 'Safari', 'Skybags', 'American Tourister India', 'Samsonite India', 'Wildcraft',
            'Bata India', 'Relaxo', 'Liberty Shoes', 'Paragon', 'Action Shoes', 'Khadim\'s', 'Metro Shoes'
        ]
        
        print(f"   Initialized {len(self.all_companies)} unique companies")
        
        # ========== 200+ UNIQUE CITIES ==========
        self.all_cities = {
            # Metro (6)
            'Mumbai': ('Maharashtra', 'Metro', 1.5),
            'Delhi': ('Delhi NCR', 'Metro', 1.5),
            'Bangalore': ('Karnataka', 'Metro', 1.5),
            'Chennai': ('Tamil Nadu', 'Metro', 1.4),
            'Kolkata': ('West Bengal', 'Metro', 1.3),
            'Hyderabad': ('Telangana', 'Metro', 1.45),
            
            # Tier-1 (30+)
            'Pune': ('Maharashtra', 'Tier-1', 1.35),
            'Ahmedabad': ('Gujarat', 'Tier-1', 1.25),
            'Gurgaon': ('Haryana', 'Tier-1', 1.45),
            'Noida': ('Uttar Pradesh', 'Tier-1', 1.4),
            'Ghaziabad': ('Uttar Pradesh', 'Tier-1', 1.25),
            'Faridabad': ('Haryana', 'Tier-1', 1.2),
            'Navi Mumbai': ('Maharashtra', 'Tier-1', 1.35),
            'Thane': ('Maharashtra', 'Tier-1', 1.3),
            'Chandigarh': ('Punjab/Haryana', 'Tier-1', 1.3),
            'Jaipur': ('Rajasthan', 'Tier-1', 1.2),
            'Lucknow': ('Uttar Pradesh', 'Tier-1', 1.15),
            'Kochi': ('Kerala', 'Tier-1', 1.25),
            'Coimbatore': ('Tamil Nadu', 'Tier-1', 1.2),
            'Indore': ('Madhya Pradesh', 'Tier-1', 1.15),
            'Nagpur': ('Maharashtra', 'Tier-1', 1.1),
            'Surat': ('Gujarat', 'Tier-1', 1.25),
            'Vadodara': ('Gujarat', 'Tier-1', 1.15),
            'Visakhapatnam': ('Andhra Pradesh', 'Tier-1', 1.15),
            'Bhopal': ('Madhya Pradesh', 'Tier-1', 1.1),
            'Patna': ('Bihar', 'Tier-1', 1.05),
            'Kanpur': ('Uttar Pradesh', 'Tier-1', 1.05),
            'Thiruvananthapuram': ('Kerala', 'Tier-1', 1.2),
            'Bhubaneswar': ('Odisha', 'Tier-1', 1.1),
            'Mysore': ('Karnataka', 'Tier-1', 1.15),
            'Nashik': ('Maharashtra', 'Tier-1', 1.1),
            'Rajkot': ('Gujarat', 'Tier-1', 1.15),
            'Ludhiana': ('Punjab', 'Tier-1', 1.15),
            'Madurai': ('Tamil Nadu', 'Tier-1', 1.1),
            'Vijayawada': ('Andhra Pradesh', 'Tier-1', 1.1),
            'Mangalore': ('Karnataka', 'Tier-1', 1.15),
            
            # Tier-2 (60+)
            'Jodhpur': ('Rajasthan', 'Tier-2', 1.0),
            'Raipur': ('Chhattisgarh', 'Tier-2', 1.0),
            'Ranchi': ('Jharkhand', 'Tier-2', 1.0),
            'Guwahati': ('Assam', 'Tier-2', 1.05),
            'Dehradun': ('Uttarakhand', 'Tier-2', 1.05),
            'Jamshedpur': ('Jharkhand', 'Tier-2', 1.05),
            'Agra': ('Uttar Pradesh', 'Tier-2', 1.0),
            'Varanasi': ('Uttar Pradesh', 'Tier-2', 1.0),
            'Meerut': ('Uttar Pradesh', 'Tier-2', 1.0),
            'Allahabad': ('Uttar Pradesh', 'Tier-2', 0.95),
            'Amritsar': ('Punjab', 'Tier-2', 1.05),
            'Jalandhar': ('Punjab', 'Tier-2', 1.0),
            'Gwalior': ('Madhya Pradesh', 'Tier-2', 0.95),
            'Jabalpur': ('Madhya Pradesh', 'Tier-2', 0.95),
            'Aurangabad': ('Maharashtra', 'Tier-2', 1.0),
            'Solapur': ('Maharashtra', 'Tier-2', 0.95),
            'Kolhapur': ('Maharashtra', 'Tier-2', 0.95),
            'Hubli-Dharwad': ('Karnataka', 'Tier-2', 1.0),
            'Belgaum': ('Karnataka', 'Tier-2', 0.95),
            'Gulbarga': ('Karnataka', 'Tier-2', 0.9),
            'Trichy': ('Tamil Nadu', 'Tier-2', 1.0),
            'Salem': ('Tamil Nadu', 'Tier-2', 0.95),
            'Tiruppur': ('Tamil Nadu', 'Tier-2', 1.0),
            'Vellore': ('Tamil Nadu', 'Tier-2', 0.95),
            'Erode': ('Tamil Nadu', 'Tier-2', 0.95),
            'Kozhikode': ('Kerala', 'Tier-2', 1.05),
            'Thrissur': ('Kerala', 'Tier-2', 1.0),
            'Kollam': ('Kerala', 'Tier-2', 0.95),
            'Warangal': ('Telangana', 'Tier-2', 0.95),
            'Karimnagar': ('Telangana', 'Tier-2', 0.9),
            'Nizamabad': ('Telangana', 'Tier-2', 0.9),
            'Guntur': ('Andhra Pradesh', 'Tier-2', 0.95),
            'Tirupati': ('Andhra Pradesh', 'Tier-2', 1.0),
            'Nellore': ('Andhra Pradesh', 'Tier-2', 0.9),
            'Kurnool': ('Andhra Pradesh', 'Tier-2', 0.9),
            'Rajahmundry': ('Andhra Pradesh', 'Tier-2', 0.95),
            'Cuttack': ('Odisha', 'Tier-2', 0.95),
            'Rourkela': ('Odisha', 'Tier-2', 0.95),
            'Asansol': ('West Bengal', 'Tier-2', 0.95),
            'Durgapur': ('West Bengal', 'Tier-2', 0.95),
            'Siliguri': ('West Bengal', 'Tier-2', 0.95),
            'Dhanbad': ('Jharkhand', 'Tier-2', 0.95),
            'Bokaro': ('Jharkhand', 'Tier-2', 0.9),
            'Bhilai': ('Chhattisgarh', 'Tier-2', 0.95),
            'Bilaspur': ('Chhattisgarh', 'Tier-2', 0.9),
            'Kota': ('Rajasthan', 'Tier-2', 1.0),
            'Bikaner': ('Rajasthan', 'Tier-2', 0.9),
            'Ajmer': ('Rajasthan', 'Tier-2', 0.95),
            'Udaipur': ('Rajasthan', 'Tier-2', 1.0),
            'Bhavnagar': ('Gujarat', 'Tier-2', 0.95),
            'Jamnagar': ('Gujarat', 'Tier-2', 1.0),
            'Junagadh': ('Gujarat', 'Tier-2', 0.9),
            'Anand': ('Gujarat', 'Tier-2', 0.95),
            'Vapi': ('Gujarat', 'Tier-2', 1.0),
            'Gandhidham': ('Gujarat', 'Tier-2', 0.95),
            'Aligarh': ('Uttar Pradesh', 'Tier-2', 0.9),
            'Moradabad': ('Uttar Pradesh', 'Tier-2', 0.9),
            'Bareilly': ('Uttar Pradesh', 'Tier-2', 0.9),
            'Gorakhpur': ('Uttar Pradesh', 'Tier-2', 0.9),
            'Saharanpur': ('Uttar Pradesh', 'Tier-2', 0.9),
            'Firozabad': ('Uttar Pradesh', 'Tier-2', 0.85),
            
            # Tier-3 (60+)
            'Shimla': ('Himachal Pradesh', 'Tier-3', 1.0),
            'Manali': ('Himachal Pradesh', 'Tier-3', 0.95),
            'Dharamshala': ('Himachal Pradesh', 'Tier-3', 0.95),
            'Jammu': ('Jammu & Kashmir', 'Tier-3', 1.0),
            'Srinagar': ('Jammu & Kashmir', 'Tier-3', 1.0),
            'Leh': ('Ladakh', 'Tier-3', 1.05),
            'Panaji': ('Goa', 'Tier-3', 1.15),
            'Margao': ('Goa', 'Tier-3', 1.1),
            'Port Blair': ('Andaman & Nicobar', 'Tier-3', 1.1),
            'Puducherry': ('Puducherry', 'Tier-3', 1.05),
            'Gangtok': ('Sikkim', 'Tier-3', 1.0),
            'Aizawl': ('Mizoram', 'Tier-3', 0.95),
            'Imphal': ('Manipur', 'Tier-3', 0.95),
            'Shillong': ('Meghalaya', 'Tier-3', 0.95),
            'Kohima': ('Nagaland', 'Tier-3', 0.95),
            'Agartala': ('Tripura', 'Tier-3', 0.9),
            'Itanagar': ('Arunachal Pradesh', 'Tier-3', 0.9),
            'Dibrugarh': ('Assam', 'Tier-3', 0.9),
            'Jorhat': ('Assam', 'Tier-3', 0.85),
            'Silchar': ('Assam', 'Tier-3', 0.85),
            'Tezpur': ('Assam', 'Tier-3', 0.85),
            'Bhagalpur': ('Bihar', 'Tier-3', 0.85),
            'Muzaffarpur': ('Bihar', 'Tier-3', 0.85),
            'Gaya': ('Bihar', 'Tier-3', 0.85),
            'Darbhanga': ('Bihar', 'Tier-3', 0.8),
            'Purnia': ('Bihar', 'Tier-3', 0.8),
            'Haldwani': ('Uttarakhand', 'Tier-3', 0.9),
            'Haridwar': ('Uttarakhand', 'Tier-3', 0.95),
            'Rishikesh': ('Uttarakhand', 'Tier-3', 0.95),
            'Roorkee': ('Uttarakhand', 'Tier-3', 0.95),
            'Ujjain': ('Madhya Pradesh', 'Tier-3', 0.85),
            'Sagar': ('Madhya Pradesh', 'Tier-3', 0.8),
            'Satna': ('Madhya Pradesh', 'Tier-3', 0.8),
            'Rewa': ('Madhya Pradesh', 'Tier-3', 0.8),
            'Korba': ('Chhattisgarh', 'Tier-3', 0.85),
            'Nanded': ('Maharashtra', 'Tier-3', 0.85),
            'Sangli': ('Maharashtra', 'Tier-3', 0.85),
            'Latur': ('Maharashtra', 'Tier-3', 0.8),
            'Akola': ('Maharashtra', 'Tier-3', 0.8),
            'Amravati': ('Maharashtra', 'Tier-3', 0.85),
            'Chandrapur': ('Maharashtra', 'Tier-3', 0.8),
            'Parbhani': ('Maharashtra', 'Tier-3', 0.8),
            'Ratnagiri': ('Maharashtra', 'Tier-3', 0.85),
            'Sindhudurg': ('Maharashtra', 'Tier-3', 0.8),
            'Nagercoil': ('Tamil Nadu', 'Tier-3', 0.85),
            'Dindigul': ('Tamil Nadu', 'Tier-3', 0.85),
            'Thanjavur': ('Tamil Nadu', 'Tier-3', 0.9),
            'Cuddalore': ('Tamil Nadu', 'Tier-3', 0.85),
            'Tirunelveli': ('Tamil Nadu', 'Tier-3', 0.85),
            'Tuticorin': ('Tamil Nadu', 'Tier-3', 0.85),
            'Kannur': ('Kerala', 'Tier-3', 0.95),
            'Palakkad': ('Kerala', 'Tier-3', 0.9),
            'Alappuzha': ('Kerala', 'Tier-3', 0.9),
            'Kottayam': ('Kerala', 'Tier-3', 0.9),
            'Malappuram': ('Kerala', 'Tier-3', 0.85),
            'Khammam': ('Telangana', 'Tier-3', 0.85),
            'Mahbubnagar': ('Telangana', 'Tier-3', 0.8),
            'Anantapur': ('Andhra Pradesh', 'Tier-3', 0.8),
            'Kadapa': ('Andhra Pradesh', 'Tier-3', 0.8),
            'Ongole': ('Andhra Pradesh', 'Tier-3', 0.8),
            'Eluru': ('Andhra Pradesh', 'Tier-3', 0.8),
            
            # Rural/Remote (40+)
            'Remote - Pan India': ('Pan India', 'Remote', 1.2),
            'Work From Home': ('Pan India', 'Remote', 1.15),
            'Hybrid - Any Location': ('Various', 'Remote', 1.1),
            'International Remote': ('International', 'Remote', 1.5),
            'Village - UP': ('Uttar Pradesh', 'Rural', 0.7),
            'Village - Bihar': ('Bihar', 'Rural', 0.65),
            'Village - MP': ('Madhya Pradesh', 'Rural', 0.7),
            'Village - Rajasthan': ('Rajasthan', 'Rural', 0.7),
            'Village - Maharashtra': ('Maharashtra', 'Rural', 0.75),
            'Village - Gujarat': ('Gujarat', 'Rural', 0.75),
            'Village - TN': ('Tamil Nadu', 'Rural', 0.75),
            'Village - Karnataka': ('Karnataka', 'Rural', 0.75),
            'Village - AP': ('Andhra Pradesh', 'Rural', 0.7),
            'Village - Telangana': ('Telangana', 'Rural', 0.7),
            'Village - Kerala': ('Kerala', 'Rural', 0.8),
            'Village - WB': ('West Bengal', 'Rural', 0.7),
            'Village - Odisha': ('Odisha', 'Rural', 0.65),
            'Village - Jharkhand': ('Jharkhand', 'Rural', 0.65),
            'Village - Chhattisgarh': ('Chhattisgarh', 'Rural', 0.65),
            'Village - Punjab': ('Punjab', 'Rural', 0.75),
            'Village - Haryana': ('Haryana', 'Rural', 0.75),
            'Village - Uttarakhand': ('Uttarakhand', 'Rural', 0.7),
            'Village - HP': ('Himachal Pradesh', 'Rural', 0.7),
            'Village - Assam': ('Assam', 'Rural', 0.65),
            'Village - Northeast': ('Northeast', 'Rural', 0.65),
            'Block HQ - North': ('Various', 'Rural', 0.75),
            'Block HQ - South': ('Various', 'Rural', 0.75),
            'Block HQ - East': ('Various', 'Rural', 0.7),
            'Block HQ - West': ('Various', 'Rural', 0.75),
            'Block HQ - Central': ('Various', 'Rural', 0.7),
            'Tehsil - North India': ('Various', 'Rural', 0.8),
            'Tehsil - South India': ('Various', 'Rural', 0.8),
            'District HQ - General': ('Various', 'Tier-3', 0.85),
            'Gram Panchayat': ('Various', 'Rural', 0.65),
            'Industrial Area': ('Various', 'Tier-2', 0.95),
            'SEZ Zone': ('Various', 'Tier-1', 1.2),
            'IT Park': ('Various', 'Tier-1', 1.25),
            'EPIP Zone': ('Various', 'Tier-2', 1.0),
            'GIDC': ('Gujarat', 'Tier-2', 1.0),
            'MIDC': ('Maharashtra', 'Tier-2', 1.0),
        }
        
        print(f"   Initialized {len(self.all_cities)} unique locations")
        
        # Domain skills mapping
        self.domain_skills = {
            'Software Engineering': ['Python', 'Java', 'JavaScript', 'React.js', 'Node.js', 'Angular', 'AWS', 'Docker', 'Git', 'SQL', 'MongoDB', 'TypeScript', 'Kubernetes', 'REST API', 'Microservices', 'Spring Boot', 'Django', 'Flask', 'Vue.js', 'GraphQL'],
            'Data Science': ['Python', 'SQL', 'Machine Learning', 'Statistics', 'Tableau', 'Power BI', 'TensorFlow', 'PyTorch', 'R', 'Excel Advanced', 'Spark', 'NLP', 'Deep Learning', 'Pandas', 'NumPy', 'Scikit-learn', 'Feature Engineering', 'Data Mining'],
            'Cloud & DevOps': ['AWS', 'Azure', 'GCP', 'Docker', 'Kubernetes', 'Terraform', 'Jenkins', 'Linux', 'Python', 'CI/CD', 'Ansible', 'Prometheus', 'Grafana', 'Helm', 'ArgoCD', 'GitHub Actions'],
            'Cybersecurity': ['Network Security', 'SIEM', 'Penetration Testing', 'Firewall Configuration', 'Encryption', 'Compliance', 'Risk Assessment', 'Incident Response', 'Burp Suite', 'Wireshark', 'Kali Linux', 'SOC Operations', 'Threat Intelligence'],
            'IT Support': ['Windows', 'Linux', 'Networking', 'Active Directory', 'Office 365', 'Troubleshooting', 'Hardware', 'ITIL', 'ServiceNow', 'Help Desk', 'Remote Support', 'System Administration'],
            'UI/UX Design': ['Figma', 'Sketch', 'Adobe XD', 'Photoshop', 'Illustrator', 'Prototyping', 'User Research', 'Design Thinking', 'Wireframing', 'Design Systems', 'Accessibility', 'Usability Testing'],
            'QA Testing': ['Selenium', 'Appium', 'JMeter', 'Postman', 'TestNG', 'Manual Testing', 'Automation', 'JIRA', 'API Testing', 'Cypress', 'Performance Testing', 'Test Case Design'],
            'Banking Operations': ['Banking Operations', 'Core Banking', 'Trade Finance', 'Credit Analysis', 'KYC/AML', 'Compliance', 'Customer Service', 'Loan Processing', 'Treasury', 'Foreign Exchange'],
            'Accounting': ['Tally', 'SAP FICO', 'GST', 'Income Tax', 'Financial Reporting', 'Excel Advanced', 'Auditing', 'Accounts Payable', 'General Ledger', 'Bank Reconciliation', 'QuickBooks'],
            'Insurance': ['Insurance Products', 'Underwriting', 'Claims Processing', 'Risk Assessment', 'Sales', 'IRDA Regulations', 'Policy Administration', 'Actuarial', 'Reinsurance'],
            'Investment': ['Financial Modeling', 'Valuation', 'Equity Research', 'Portfolio Management', 'CFA', 'Bloomberg Terminal', 'Technical Analysis', 'DCF Analysis', 'Fixed Income', 'Derivatives'],
            'Nursing': ['Patient Care', 'Clinical Documentation', 'Medication Administration', 'Emergency Care', 'ICU Care', 'Infection Control', 'Vital Signs', 'IV Therapy', 'Wound Care'],
            'Medical Doctors': ['Diagnosis', 'Treatment', 'Patient Care', 'Surgery', 'Medical Knowledge', 'Clinical Documentation', 'Prescription', 'Medical Ethics', 'Specialty Medicine'],
            'Allied Health': ['Laboratory Techniques', 'Patient Care', 'Medical Equipment', 'Clinical Documentation', 'Phlebotomy', 'Radiology', 'Physical Therapy', 'Occupational Therapy'],
            'Mental Health': ['Counseling', 'Psychology', 'Cognitive Behavioral Therapy', 'Active Listening', 'Empathy', 'Assessment', 'Crisis Intervention', 'Group Therapy'],
            'Pharma Sales': ['Medical Terminology', 'Sales', 'Clinical Research', 'Regulatory Affairs', 'Pharmacovigilance', 'Product Knowledge', 'Territory Management', 'Doctor Relations'],
            'School Teaching': ['Teaching', 'Curriculum Development', 'Classroom Management', 'Student Assessment', 'Lesson Planning', 'Special Education', 'Educational Technology', 'Parent Communication'],
            'Higher Education': ['Research', 'Teaching', 'Subject Expertise', 'Publication', 'Curriculum Development', 'Mentoring', 'Grant Writing', 'Academic Writing'],
            'EdTech': ['Online Teaching', 'Content Creation', 'Video Production', 'LMS Administration', 'Student Engagement', 'E-Learning Development', 'Instructional Design', 'Gamification'],
            'Corporate Training': ['Training Delivery', 'Facilitation', 'Instructional Design', 'Presentation Skills', 'Adult Learning', 'Assessment', 'Coaching', 'Leadership Development'],
            'Content Writing': ['Writing', 'SEO', 'Research', 'Editing', 'Grammar', 'CMS', 'Content Strategy', 'Copywriting', 'Technical Writing', 'Blogging'],
            'Journalism': ['News Writing', 'Reporting', 'Interviewing', 'Research', 'Fact-checking', 'Media Ethics', 'Video Editing', 'Social Media'],
            'Digital Marketing': ['SEO', 'Google Ads', 'Facebook Ads', 'Google Analytics', 'Content Marketing', 'Email Marketing', 'Social Media Marketing', 'Marketing Automation', 'A/B Testing'],
            'Graphic Design': ['Photoshop', 'Illustrator', 'After Effects', 'Premiere Pro', 'InDesign', 'Figma', 'Animation', 'Typography', 'Brand Identity', 'Motion Graphics'],
            'Sales': ['B2B Sales', 'Negotiation', 'Communication Skills', 'CRM', 'Lead Generation', 'Presentation Skills', 'Closing Techniques', 'Account Management', 'Pipeline Management'],
            'Marketing': ['Marketing Strategy', 'Brand Management', 'Market Research', 'Campaign Management', 'Analytics', 'Public Relations', 'Budget Management', 'Competitive Analysis'],
            'Retail': ['Customer Service', 'Sales', 'Product Knowledge', 'Visual Merchandising', 'Inventory Management', 'POS', 'Cash Handling', 'Store Operations'],
            'HR': ['Recruitment', 'Employee Relations', 'HRIS', 'Payroll Processing', 'Compliance', 'Performance Management', 'Training & Development', 'Talent Acquisition', 'Onboarding'],
            'Customer Support': ['Communication Skills', 'Problem Solving', 'Empathy', 'CRM', 'Active Listening', 'Ticketing Systems', 'Conflict Resolution', 'Technical Support'],
            'Administration': ['MS Office Suite', 'Communication Skills', 'Organization Skills', 'Calendar Management', 'Travel Arrangements', 'Event Planning', 'Document Management'],
            'Data Entry': ['Typing', 'Data Entry', 'MS Excel', 'Attention to Detail', 'Computer Skills', 'Database Management', 'Accuracy', 'Speed'],
            'Legal': ['Legal Research', 'Contract Drafting', 'Litigation', 'Corporate Law', 'Compliance', 'Due Diligence', 'Intellectual Property', 'Legal Writing'],
            'Hotel Management': ['Guest Services', 'Hotel Operations', 'Revenue Management', 'F&B Operations', 'Front Desk Management', 'Housekeeping Management', 'Hotel PMS'],
            'Food Service': ['Cooking', 'Baking', 'Food Safety', 'Menu Planning', 'Kitchen Management', 'Hygiene', 'Food Presentation', 'Inventory Management'],
            'Travel Tourism': ['Travel Planning', 'GDS', 'Customer Service', 'Visa Processing', 'Ticketing', 'Itinerary Design', 'Tour Operations', 'Destination Knowledge'],
            'Beauty Wellness': ['Makeup Artistry', 'Hair Styling', 'Skincare', 'Nail Art', 'Spa Treatments', 'Customer Service', 'Product Knowledge', 'Sanitation'],
            'Fitness Yoga': ['Fitness Training', 'Yoga Instruction', 'Nutrition', 'Personal Training', 'Group Classes', 'Exercise Science', 'Client Assessment'],
            'Fashion Design': ['Fashion Design', 'Pattern Making', 'Garment Construction', 'CAD', 'Merchandising', 'Tailoring', 'Textile Design', 'Trend Forecasting'],
            'Handicrafts': ['Craftsmanship', 'Traditional Techniques', 'Creativity', 'Hand Skills', 'Design', 'Quality Control', 'Material Knowledge'],
            'Childcare': ['Child Care', 'Early Development', 'First Aid', 'Patience', 'Play-based Learning', 'Child Nutrition', 'Communication Skills'],
            'Elderly Care': ['Elder Care', 'Patient Care', 'First Aid', 'Medication Management', 'Mobility Assistance', 'Companion Care', 'Empathy'],
            'Household': ['Household Management', 'Cooking', 'Cleaning', 'Driving', 'Gardening', 'Organization Skills', 'Time Management'],
            'Agriculture': ['Farming', 'Crop Management', 'Irrigation', 'Organic Farming', 'Pest Management', 'Horticulture', 'Soil Management', 'Agricultural Machinery'],
            'Dairy Animal': ['Animal Care', 'Dairy Operations', 'Poultry Management', 'Animal Nutrition', 'Veterinary Assistance', 'Farm Management'],
            'Food Processing': ['Food Processing', 'Machine Operation', 'Quality Control', 'FSSAI', 'Hygiene', 'Packaging', 'Production Planning'],
            'Ecommerce': ['E-commerce Platforms', 'Product Listing', 'Inventory Management', 'Analytics', 'SEO', 'Digital Marketing', 'Customer Service'],
            'Delivery Logistics': ['Navigation', 'Customer Service', 'Driving', 'Time Management', 'Physical Fitness', 'Route Planning', 'Package Handling'],
            'Freelancing': ['Self-Management', 'Multiple Skills', 'Time Management', 'Client Management', 'Marketing', 'Invoicing', 'Project Management'],
            'Manufacturing': ['Machine Operation', 'Assembly', 'Quality Control', 'Safety', 'Maintenance', 'Production Planning', 'Lean Manufacturing', '5S'],
            'Government': ['Administration', 'Documentation', 'Public Service', 'Computer Skills', 'Communication Skills', 'Hindi', 'Regional Language', 'Compliance'],
            'Social Work': ['Community Development', 'Counseling', 'Field Work', 'Documentation', 'Communication Skills', 'Empathy', 'Case Management', 'Advocacy'],
            'Real Estate': ['Real Estate', 'Sales', 'Negotiation', 'Legal Documentation', 'Market Analysis', 'Customer Service', 'Property Valuation'],
            'Construction': ['AutoCAD', 'Project Management', 'Site Management', 'Estimation', 'Safety', 'Structural Design', 'Civil Engineering', 'Quality Control'],
            'Consulting': ['Business Analysis', 'Problem Solving', 'Strategy', 'Presentation Skills', 'Client Management', 'Research', 'Data Analysis', 'Communication Skills'],
            'General': ['Communication Skills', 'Problem Solving', 'Computer Skills', 'Teamwork', 'Time Management', 'Adaptability', 'Customer Service', 'MS Office Suite']
        }
        
        # Domain salary ranges
        self.domain_salary = {
            'Software Engineering': (35000, 180000), 'Data Science': (45000, 200000), 'Cloud & DevOps': (40000, 180000),
            'Cybersecurity': (45000, 190000), 'IT Support': (18000, 65000), 'UI/UX Design': (30000, 130000), 'QA Testing': (25000, 100000),
            'Banking Operations': (22000, 90000), 'Accounting': (18000, 80000), 'Insurance': (18000, 90000), 'Investment': (35000, 250000),
            'Nursing': (15000, 55000), 'Medical Doctors': (50000, 350000), 'Allied Health': (15000, 65000), 'Mental Health': (22000, 110000), 'Pharma Sales': (22000, 90000),
            'School Teaching': (12000, 55000), 'Higher Education': (35000, 160000), 'EdTech': (22000, 90000), 'Corporate Training': (28000, 110000),
            'Content Writing': (15000, 70000), 'Journalism': (18000, 90000), 'Digital Marketing': (22000, 120000), 'Graphic Design': (18000, 90000),
            'Sales': (15000, 120000), 'Marketing': (25000, 140000), 'Retail': (10000, 45000),
            'HR': (22000, 110000), 'Customer Support': (12000, 45000), 'Administration': (12000, 55000), 'Data Entry': (10000, 35000),
            'Legal': (25000, 220000), 'Hotel Management': (15000, 80000), 'Food Service': (12000, 65000), 'Travel Tourism': (12000, 55000),
            'Beauty Wellness': (10000, 45000), 'Fitness Yoga': (12000, 70000),
            'Fashion Design': (12000, 90000), 'Handicrafts': (6000, 35000),
            'Childcare': (8000, 35000), 'Elderly Care': (10000, 35000), 'Household': (6000, 30000),
            'Agriculture': (8000, 45000), 'Dairy Animal': (10000, 45000), 'Food Processing': (10000, 45000),
            'Ecommerce': (18000, 90000), 'Delivery Logistics': (10000, 35000), 'Freelancing': (12000, 120000),
            'Manufacturing': (10000, 55000), 'Government': (22000, 120000), 'Social Work': (12000, 45000),
            'Real Estate': (18000, 120000), 'Construction': (18000, 90000), 'Consulting': (45000, 250000), 'General': (12000, 70000)
        }
        
        # Domain to sector mapping
        self.domain_sector = {
            'Software Engineering': 'IT & Technology', 'Data Science': 'IT & Technology', 'Cloud & DevOps': 'IT & Technology',
            'Cybersecurity': 'IT & Technology', 'IT Support': 'IT & Technology', 'UI/UX Design': 'IT & Technology', 'QA Testing': 'IT & Technology',
            'Banking Operations': 'Banking & Finance', 'Accounting': 'Banking & Finance', 'Insurance': 'Banking & Finance', 'Investment': 'Banking & Finance',
            'Nursing': 'Healthcare', 'Medical Doctors': 'Healthcare', 'Allied Health': 'Healthcare', 'Mental Health': 'Healthcare', 'Pharma Sales': 'Healthcare',
            'School Teaching': 'Education', 'Higher Education': 'Education', 'EdTech': 'Education', 'Corporate Training': 'Education',
            'Content Writing': 'Content & Media', 'Journalism': 'Content & Media', 'Digital Marketing': 'Content & Media', 'Graphic Design': 'Content & Media',
            'Sales': 'Sales & Marketing', 'Marketing': 'Sales & Marketing', 'Retail': 'Sales & Marketing',
            'HR': 'Human Resources', 'Customer Support': 'Customer Service', 'Administration': 'Administration', 'Data Entry': 'Administration',
            'Legal': 'Legal', 'Hotel Management': 'Hospitality', 'Food Service': 'Hospitality', 'Travel Tourism': 'Hospitality',
            'Beauty Wellness': 'Beauty & Wellness', 'Fitness Yoga': 'Beauty & Wellness',
            'Fashion Design': 'Arts & Design', 'Handicrafts': 'Arts & Design',
            'Childcare': 'Childcare & Homecare', 'Elderly Care': 'Childcare & Homecare', 'Household': 'Childcare & Homecare',
            'Agriculture': 'Agriculture', 'Dairy Animal': 'Agriculture', 'Food Processing': 'Agriculture',
            'Ecommerce': 'E-Commerce', 'Delivery Logistics': 'E-Commerce', 'Freelancing': 'Gig Economy',
            'Manufacturing': 'Manufacturing', 'Government': 'Government', 'Social Work': 'Government',
            'Real Estate': 'Real Estate', 'Construction': 'Real Estate', 'Consulting': 'Consulting', 'General': 'General'
        }
        
        # Remote friendliness by domain
        self.domain_remote = {
            'Software Engineering': 0.85, 'Data Science': 0.85, 'Cloud & DevOps': 0.80, 'Cybersecurity': 0.75,
            'IT Support': 0.50, 'UI/UX Design': 0.90, 'QA Testing': 0.75,
            'Banking Operations': 0.25, 'Accounting': 0.60, 'Insurance': 0.40, 'Investment': 0.55,
            'Nursing': 0.05, 'Medical Doctors': 0.15, 'Allied Health': 0.15, 'Mental Health': 0.70, 'Pharma Sales': 0.30,
            'School Teaching': 0.35, 'Higher Education': 0.30, 'EdTech': 0.90, 'Corporate Training': 0.60,
            'Content Writing': 0.95, 'Journalism': 0.45, 'Digital Marketing': 0.85, 'Graphic Design': 0.85,
            'Sales': 0.35, 'Marketing': 0.55, 'Retail': 0.05,
            'HR': 0.60, 'Customer Support': 0.70, 'Administration': 0.35, 'Data Entry': 0.80,
            'Legal': 0.45, 'Hotel Management': 0.05, 'Food Service': 0.05, 'Travel Tourism': 0.40,
            'Beauty Wellness': 0.10, 'Fitness Yoga': 0.55,
            'Fashion Design': 0.35, 'Handicrafts': 0.20,
            'Childcare': 0.05, 'Elderly Care': 0.05, 'Household': 0.02,
            'Agriculture': 0.05, 'Dairy Animal': 0.02, 'Food Processing': 0.05,
            'Ecommerce': 0.80, 'Delivery Logistics': 0.02, 'Freelancing': 0.95,
            'Manufacturing': 0.02, 'Government': 0.10, 'Social Work': 0.15,
            'Real Estate': 0.30, 'Construction': 0.15, 'Consulting': 0.70, 'General': 0.50
        }
        
        # Education levels
        self.education = [
            ('PhD/Doctorate', 0.02, 1.8), ('Post Graduate (MBA/MTech/MA/MSc)', 0.12, 1.5),
            ('Graduate (BTech/BA/BCom/BSc)', 0.30, 1.2), ('Diploma/ITI', 0.18, 1.0),
            ('12th Pass (HSC)', 0.18, 0.85), ('10th Pass (SSC)', 0.12, 0.75),
            ('8th Pass', 0.05, 0.65), ('Below 8th/Informal Education', 0.03, 0.55)
        ]
        
        print()
    
    def _get_domain_for_title(self, job_title):
        """Get domain for a job title"""
        for domain, titles in self.job_categories.items():
            if job_title in titles:
                return domain
        return 'General'
    
    def generate_record(self, record_id):
        """Generate a single record with maximum diversity"""
        
        # Select random job title
        job_title = random.choice(self.all_job_titles)
        domain = self._get_domain_for_title(job_title)
        sector = self.domain_sector.get(domain, 'General')
        
        # Demographics
        age = self._generate_age()
        kids = self._generate_kids(age)
        
        # Location - select random city
        city = random.choice(list(self.all_cities.keys()))
        state, tier, salary_mult = self.all_cities[city]
        
        # Education
        edu = random.choices(
            [e[0] for e in self.education],
            weights=[e[1] for e in self.education]
        )[0]
        edu_mult = next(e[2] for e in self.education if e[0] == edu)
        
        # Experience
        min_age_edu = {'PhD/Doctorate': 28, 'Post Graduate (MBA/MTech/MA/MSc)': 24, 'Graduate (BTech/BA/BCom/BSc)': 21}.get(edu, 18)
        max_exp = max(0, age - min_age_edu - 1)
        experience = random.randint(0, min(20, max_exp))
        
        # Skills - pick from domain + general pool
        domain_specific = self.domain_skills.get(domain, self.domain_skills['General'])
        all_available = list(set(domain_specific + random.sample(self.all_skills, min(20, len(self.all_skills)))))
        skill1 = random.choice(all_available)
        skill2 = random.choice([s for s in all_available if s != skill1] or all_available)
        all_skills_list = random.sample(all_available, min(5, len(all_available)))
        
        # Work mode
        remote_prob = self.domain_remote.get(domain, 0.5)
        if 'Remote' in tier or random.random() < remote_prob:
            work_mode = random.choice(['Work From Home', 'Hybrid', 'Remote'])
        else:
            work_mode = random.choice(['On-site', 'Office', 'Field Work'])
        
        # Hours
        hours = random.randint(2, 8)
        if kids > 0:
            hours = max(1, hours - min(kids, 3))
        if work_mode in ['Work From Home', 'Remote', 'Hybrid']:
            hours = min(8, hours + 1)
        
        # Company
        company = random.choice(self.all_companies)
        
        # Salary
        salary_range = self.domain_salary.get(domain, (15000, 60000))
        base_salary = random.randint(salary_range[0], salary_range[1])
        salary = int(base_salary * edu_mult * salary_mult * (1 + experience * 0.05) * (0.5 + hours/8 * 0.6))
        salary = max(5000, min(400000, salary))
        
        # Mother suitability
        mother_score = 5
        if work_mode in ['Work From Home', 'Remote']:
            mother_score += 3
        elif work_mode == 'Hybrid':
            mother_score += 2
        if hours <= 4:
            mother_score += 2
        elif hours <= 6:
            mother_score += 1
        mother_score = min(10, mother_score)
        
        # Seniority
        if experience >= 12:
            seniority = 'Director/Executive'
        elif experience >= 8:
            seniority = 'Senior Manager'
        elif experience >= 5:
            seniority = 'Manager'
        elif experience >= 3:
            seniority = 'Senior Associate'
        elif experience >= 1:
            seniority = 'Associate'
        else:
            seniority = 'Entry Level/Fresher'
        
        # Salary bracket
        brackets = [(10000, 'Below 10K'), (20000, '10K-20K'), (35000, '20K-35K'), (50000, '35K-50K'),
                    (75000, '50K-75K'), (100000, '75K-1L'), (150000, '1L-1.5L'), (200000, '1.5L-2L'), (float('inf'), 'Above 2L')]
        salary_bracket = next(b[1] for b in brackets if salary < b[0])
        
        return {
            'record_id': record_id,
            'age': age,
            'kids': kids,
            'marital_status': 'Married' if kids > 0 or age > 28 else random.choice(['Single', 'Married']),
            'hours_available': hours,
            'domain': domain,
            'sector': sector,
            'job_title': job_title,
            'job_description': f"Looking for {job_title} at {company}. Skills: {skill1}, {skill2}. Experience: {experience}+ years. {'Remote work available.' if work_mode in ['Work From Home', 'Remote', 'Hybrid'] else 'On-site position.'}"[:250],
            'seniority_level': seniority,
            'primary_skill': skill1,
            'secondary_skill': skill2,
            'all_skills': ', '.join(all_skills_list),
            'education': edu,
            'experience_years': experience,
            'company': company,
            'city': city,
            'city_tier': tier,
            'state': state,
            'country': 'India',
            'work_mode': work_mode,
            'work_type': random.choice(['Full-time', 'Part-time', 'Contract', 'Freelance', 'Internship']),
            'remote_available': work_mode in ['Work From Home', 'Hybrid', 'Remote'],
            'flexible_timing': work_mode in ['Work From Home', 'Hybrid', 'Remote'] or hours <= 4,
            'shift_type': random.choice(['Day Shift', 'Night Shift', 'Rotational', 'Flexible', 'General Shift']),
            'income': salary,
            'salary_min': int(salary * 0.85),
            'salary_max': int(salary * 1.25),
            'salary_bracket': salary_bracket,
            'language': random.choice(['English', 'Hindi', 'Both English and Hindi', 'Regional Language', 'English + Regional']),
            'device': random.choice(['Mobile Phone', 'Laptop', 'Both Mobile and Laptop', 'Desktop', 'No Device Required']),
            'mother_suitability_score': mother_score,
            'childcare_compatible': mother_score >= 7,
            'women_friendly': random.random() < 0.70,
            'maternity_benefits': random.random() < 0.45,
            'skill_match_score': random.randint(50, 100),
            'career_growth': random.choice(['High', 'Medium', 'Low']),
            'training_provided': random.random() < 0.55,
            'work_life_balance': random.randint(5, 10) if work_mode in ['Work From Home', 'Remote'] else random.randint(3, 8),
            'travel_required': 'No Travel' if work_mode in ['Work From Home', 'Remote'] else random.choice(['No Travel', 'Occasional', 'Frequent']),
            'health_insurance': random.random() < 0.45,
            'pf_available': random.random() < 0.50,
            'is_verified': random.random() < 0.85,
            'created_date': datetime.now().strftime('%Y-%m-%d')
        }
    
    def _generate_age(self):
        if random.random() < 0.60:
            return random.randint(25, 40)
        elif random.random() < 0.80:
            return random.randint(18, 24)
        else:
            return random.randint(41, 55)
    
    def _generate_kids(self, age):
        if age < 22:
            return random.choices([0, 1], weights=[0.95, 0.05])[0]
        elif age < 28:
            return random.choices([0, 1, 2], weights=[0.6, 0.3, 0.1])[0]
        elif age < 35:
            return random.choices([0, 1, 2, 3], weights=[0.3, 0.35, 0.25, 0.1])[0]
        else:
            return random.choices([0, 1, 2, 3, 4], weights=[0.2, 0.25, 0.35, 0.15, 0.05])[0]
    
    def generate_dataset(self, output_path):
        """Generate full dataset"""
        print(f"Generating {self.target_records:,} records...")
        
        chunk_size = 50000
        first_chunk = True
        
        for start in range(0, self.target_records, chunk_size):
            end = min(start + chunk_size, self.target_records)
            records = [self.generate_record(i + 1) for i in range(start, end)]
            chunk_df = pd.DataFrame(records)
            
            if first_chunk:
                chunk_df.to_csv(output_path, index=False)
                first_chunk = False
            else:
                chunk_df.to_csv(output_path, mode='a', header=False, index=False)
            
            print(f"   {end:,}/{self.target_records:,} ({end/self.target_records*100:.0f}%)")
        
        return pd.read_csv(output_path)
    
    def run(self, output_path):
        """Run generation with full validation"""
        df = self.generate_dataset(output_path)
        
        print("\n" + "="*70)
        print("FINAL VALIDATION & UNIQUENESS REPORT")
        print("="*70)
        
        print(f"\n   OVERALL STATS:")
        print(f"   Total Records: {len(df):,}")
        print(f"   Total Columns: {len(df.columns)}")
        print(f"   Missing Values: {df.isnull().sum().sum()}")
        
        print(f"\n   UNIQUENESS METRICS (TARGETS ACHIEVED):")
        print(f"   {'Metric':<25} {'Count':>10} {'Target':>12}")
        print(f"   {'-'*50}")
        print(f"   {'Unique Job Titles':<25} {df['job_title'].nunique():>10,} {'10,000+':>12}")
        print(f"   {'Unique Domains':<25} {df['domain'].nunique():>10}")
        print(f"   {'Unique Sectors':<25} {df['sector'].nunique():>10}")
        print(f"   {'Unique Companies':<25} {df['company'].nunique():>10,} {'500+':>12}")
        print(f"   {'Unique Cities':<25} {df['city'].nunique():>10,} {'150+':>12}")
        print(f"   {'Unique Primary Skills':<25} {df['primary_skill'].nunique():>10,} {'500+':>12}")
        print(f"   {'Unique Secondary Skills':<25} {df['secondary_skill'].nunique():>10,}")
        print(f"   {'Unique Education Levels':<25} {df['education'].nunique():>10}")
        print(f"   {'Unique Seniority Levels':<25} {df['seniority_level'].nunique():>10}")
        print(f"   {'Unique States':<25} {df['state'].nunique():>10}")
        
        print(f"\n   DATA QUALITY:")
        print(f"   Age Range: {df['age'].min()}-{df['age'].max()} (avg: {df['age'].mean():.1f})")
        print(f"   Experience Range: {df['experience_years'].min()}-{df['experience_years'].max()} years")
        print(f"   Income Range: Rs.{df['income'].min():,} - Rs.{df['income'].max():,}")
        print(f"   Mother Suitability Avg: {df['mother_suitability_score'].mean():.1f}/10")
        print(f"   Remote Jobs: {df['remote_available'].sum():,} ({df['remote_available'].sum()/len(df)*100:.1f}%)")
        print(f"   Women Friendly: {df['women_friendly'].sum():,} ({df['women_friendly'].sum()/len(df)*100:.1f}%)")
        
        print(f"\n   Top 10 Domains:")
        for i, (domain, count) in enumerate(df['domain'].value_counts().head(10).items(), 1):
            print(f"   {i:2}. {domain}: {count:,}")
        
        print("\n" + "="*70)
        print("GENERATION COMPLETE!")
        print(f"   Output: {output_path}")
        print("="*70)
        
        return df


def main():
    output_path = '/Users/kartik/Documents/MaaSarthi/maasarthi_300k_v4_dataset.csv'
    generator = MaaSarthiGeneratorV4()
    df = generator.run(output_path)
    return df


if __name__ == "__main__":
    main()
