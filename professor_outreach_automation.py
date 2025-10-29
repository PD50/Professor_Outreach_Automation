#!/usr/bin/env python3
"""
Professor Outreach Automation System
Automates personalized email generation for research internship applications
"""

import os
import csv
import re
import time
import json
import base64
import mimetypes
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import pickle
import requests
import urllib3
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, unquote

# Google API imports
try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    GMAIL_AVAILABLE = True
except ImportError:
    GMAIL_AVAILABLE = False
    print("⚠️  Gmail API libraries not installed. Install with:")
    print("   pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client --break-system-packages")

# Configuration
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent"

# Resume content as structured data
RESUME_SUMMARY = """
# Parth Dawar - Resume Summary

## Contact Information
- Name: Parth Dawar
- Email: bm23btech11018@iith.ac.in
- Phone: +919814433323
- Current Institution: IIT Hyderabad, India (3rd year, 5th semester)
- Degree: B.Tech in Biomedical Engineering (Expected 2027)
- Double Major in Engineering Physics (Expected 2027)
- Minor in Economics (Expected 2027)
- CGPA: 8.90/10 (Top 5% of class, equivalent to ~9.0/10)

## Academic Background
- B.Tech in Biomedical Engineering, IIT Hyderabad (2027) - CGPA: 8.90
- Double Major in Engineering Physics, IIT Hyderabad (2027)
- Minor in Economics, IIT Hyderabad (2027)
- Class Ranking: Top 5%

## Research Experience
**Undergraduate Researcher (Remote) | DISP Lab, INSA Lyon (January 2025 - May 2025)**
- Remote research internship focused on medical image annotation
- Developing robust semi-supervised image annotation approach for fundus and angiography datasets
- Reduced manual annotation time by 40%, improved accuracy from 75% to 90%
- Working with 5,000+ image eye-imaging dataset using OpenCV and preprocessing pipelines

## Key Technical Projects

### AI-Powered Smart Prosthetic Hand
- Developed myoelectric hand translating EMG muscle signals into real-time gestures
- 3D-printed robotic hand with Arduino-controlled servo motors
- Neural network classifier using PyTorch achieving 70% accuracy for 5 distinct gestures

### Medical Image Segmentation: Brain Tumor Detection
- Developed deep learning model for automated brain tumor segmentation from MRI scans
- Implemented U-Net architecture using TensorFlow and Keras
- Achieved 92% Dice similarity coefficient for clinical diagnostic support

### AI-Assisted Robotic Ultrasound for Autonomous Medical Imaging
- Developed autonomous ultrasound system with collaborative robotic arm
- Implemented CNN and RL models for image quality optimization
- Integrated force-torque sensing with ROS motion planning

### Probabilistic NLP Framework for Airline Review Analysis
- Extracted quantifiable ratings from airline reviews using custom probabilistic NLP model
- Built probabilistic model for confidence metrics and travel itinerary suggestions
- Analyzed thousands of customer feedback entries

## Technical Skills
**Programming Languages:** Python (PyTorch, Pandas, TensorFlow, NumPy, Stable Baselines), C++/C, MATLAB, JavaScript, TypeScript

**Specialized Skills:** 
- Computer Vision
- Machine Learning & Deep Learning
- Reinforcement Learning
- Data Science & Analysis
- Medical Image Processing
- Embedded Systems

**Coursework:**
- Data Structures and Algorithms
- Machine Learning for Process System Engineering
- Biomedical Imaging
- Foundation of Machine Learning
- Linear Algebra
- Probability and Random Variables
- Mathematical Modelling and Systems Biology
- Statistics
- Analog and Integrated Circuits
- Introduction to Embedded Systems

## Notable Achievements
- Rated Expert (1646) on Codeforces, ranked 552 in Round 1028 (Div. 2)
- Received BUILD Grant worth 40,000 INR for ECG Smart Patch development
- Represented IIT Hyderabad in Inter IIT Tech Meet 12.0 (WorldQuant BRAIN)
- Represented IIT Hyderabad in Inter IIT Cultural Meet 7.0 (Quiz Club)
- Chess enthusiast (Rated 1600 on chess.com)

## Previous Industry Experience
**WorldQuant BRAIN | Inter IIT Tech Meet 12.0 (November 2023 - December 2023)**
- Developed, backtested, and optimized trading alphas for financial market predictions
- Automated alpha optimization using Python tools interfacing with BRAIN API

## Strengths for Research
- Strong foundation in biomedical engineering with computational focus
- Experience with medical imaging (MRI, ultrasound, fundus imaging)
- Proficient in deep learning frameworks (PyTorch, TensorFlow)
- Computer vision and image processing expertise
- Mathematical modeling and systems biology knowledge
- Interdisciplinary background bridging engineering and biology
- Current research experience in medical image annotation (INSA Lyon)
"""


@dataclass
class ProfessorInfo:
    """Data structure for professor information"""
    name: str
    university: str
    profile_url: str
    email: Optional[str] = None
    research_areas: List[str] = None
    department: Optional[str] = None
    full_profile_text: Optional[str] = None
    contact_page_url: Optional[str] = None


class ProfessorScraper:
    """Scrapes professor information from university websites"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        # Disable SSL verification warnings
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    def scrape_professor_page(self, url: str) -> Tuple[str, List[str], str, Optional[str]]:
        """
        Scrapes professor profile page for research information
        Returns: (full_text, research_areas, department, contact_url)
        """
        try:
            response = self.session.get(url, timeout=15, verify=False)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract all text content
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            text = soup.get_text(separator=' ', strip=True)
            text = ' '.join(text.split())  # Normalize whitespace
            
            # Extract research areas (look for common patterns)
            research_areas = []
            research_keywords = [
                'research interests', 'research areas', 'research focus',
                'research topics', 'scientific interests', 'areas of expertise'
            ]
            
            for keyword in research_keywords:
                pattern = re.compile(rf'{keyword}[:\s]+(.*?)(?:\.|Research|Publications|Teaching|Contact)', 
                                   re.IGNORECASE | re.DOTALL)
                match = pattern.search(text)
                if match:
                    areas_text = match.group(1).strip()
                    # Split by common delimiters
                    areas = re.split(r'[,;•\n]', areas_text)
                    research_areas.extend([area.strip() for area in areas if len(area.strip()) > 5])
                    break
            
            # Extract department
            department = None
            dept_pattern = re.compile(r'(?:Department|Faculty|Chair|Institute) of ([\w\s]+)', re.IGNORECASE)
            dept_match = dept_pattern.search(text)
            if dept_match:
                department = dept_match.group(1).strip()
            
            # Find contact page URL
            contact_url = None
            
            # --- MODIFICATION START ---
            # Expanded keyword list to find business cards / vCards based on your example
            contact_keywords = [
                'contact', 'email', 'reach', 
                'business card', 'contact details', 
                'visitenkarte', 'vcard', 'tumonline'
            ]
            # --- MODIFICATION END ---

            for link in soup.find_all('a', href=True):
                link_text = link.get_text().lower().strip()
                
                # --- MODIFICATION START ---
                # Check if the link text contains any of our new keywords
                if any(keyword in link_text for keyword in contact_keywords):
                # --- MODIFICATION END ---
                    
                    potential_url = urljoin(url, link['href'])
                    
                    # If the link is a 'mailto:', we don't need it here.
                    # The extract_email_from_page method will find it.
                    if potential_url.lower().startswith('mailto:'):
                        continue
                    
                    # Avoid looping back to the same page
                    if potential_url != url:
                        contact_url = potential_url
                        break # Found a good candidate
            
            return text[:3000], research_areas[:5], department, contact_url
            
        except Exception as e:
            print(f"Error scraping {url}: {str(e)}")
            return "", [], None, None
    
    def extract_email_from_page(self, url: str) -> Optional[str]:
        """Extracts email address from a webpage, ignoring generic/blacklisted emails"""
        
        # --- MODIFICATION START ---
        # Blacklist of common generic/junk email addresses to ignore
        EMAIL_BLACKLIST = [
            'example.com', 'domain.com', 'webmaster@', 'info@', 'admin@', 'test@',
            'support@', 'kontakt@', 'help@', 'no-reply@', 'noreply@',
            'professorenprofile@'  # Adding the specific one you found
        ]
        # --- MODIFICATION END ---

        try:
            response = self.session.get(url, timeout=15, verify=False)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            good_emails = []

            # Look for mailto links first
            mailto_links = soup.find_all('a', href=re.compile(r'^mailto:', re.IGNORECASE))
            for link in mailto_links:
                email = link['href'].replace('mailto:', '').split('?')[0].strip()
                # Check if it's a good email
                if email and not any(blacklisted in email.lower() for blacklisted in EMAIL_BLACKLIST):
                    good_emails.append(email)
            
            # If we found a good mailto link, return it
            if good_emails:
                return good_emails[0]

            # If no good mailto, search in text content using regex
            text = soup.get_text()
            email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
            emails = email_pattern.findall(text)
            
            if emails:
                # Filter out common non-personal emails
                for email in emails:
                    if email and not any(blacklisted in email.lower() for blacklisted in EMAIL_BLACKLIST):
                        good_emails.append(email)

            # Return the first good email found, or None if only generic ones
            return good_emails[0] if good_emails else None
            
        except Exception as e:
            print(f"Error extracting email from {url}: {str(e)}")
            return None
    
    def get_professor_info(self, profile_url: str, university: str) -> ProfessorInfo:
        """Gathers complete information about a professor"""
        print(f"Scraping: {profile_url}")
        
        # Extract professor name from URL
        name_part = profile_url.rstrip('/').split('/')[-1]
        # Handle formats like "amy-zavatsky-2" -> "Amy Zavatsky"
        name = name_part.rsplit('-', 1)[0].replace('-', ' ').title() if name_part[-1].isdigit() else name_part.replace('-', ' ').title()
        
        # Scrape profile page
        full_text, research_areas, department, contact_url = self.scrape_professor_page(profile_url)
        
        # --- MODIFICATION START: Priority Logic Changed ---
        email = None
        
        # 1. Try the contact page FIRST (e.g., the "Business card" link)
        if contact_url:
            print(f"  Checking contact page: {contact_url}")
            email = self.extract_email_from_page(contact_url)
        
        # 2. If no email on contact page, try the main profile page
        if not email:
            print("  Contact page had no email, checking main page...")
            email = self.extract_email_from_page(profile_url)
        
        # 3. If STILL no email, try searching the full text as a last resort
        #    (This finds emails not in 'mailto' links, but also filters them)
        if not email:
            print("  No linkable email found, searching page text...")
            email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
            found_emails = email_pattern.findall(full_text)
            
            EMAIL_BLACKLIST = [
                'example.com', 'domain.com', 'webmaster@', 'info@', 'admin@', 'test@',
                'support@', 'kontakt@', 'help@', 'no-reply@', 'noreply@',
                'professorenprofile@'
            ]

            if found_emails:
                for found_email in found_emails:
                    if found_email and not any(blacklisted in found_email.lower() for blacklisted in EMAIL_BLACKLIST):
                        email = found_email
                        break # Found the first good one
        # --- MODIFICATION END ---
        
        print(f"  Found: {name} | Email: {email if email else 'NOT FOUND'}")
        
        return ProfessorInfo(
            name=name,
            university=university,
            profile_url=profile_url,
            email=email,
            research_areas=research_areas if research_areas else [],
            department=department,
            full_profile_text=full_text,
            contact_page_url=contact_url
        )

class EmailGenerator:
    """Generates personalized emails using Google Gemini API"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.last_request_time = 0
        self.min_delay = 4  # Minimum 4 seconds between requests
    
    def call_gemini(self, prompt: str, max_retries: int = 10) -> str:
        """Makes API call to Google Gemini with aggressive rate limiting"""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={self.api_key}"
        
        headers = {
            "Content-Type": "application/json"
        }
        
        payload = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 2048,
            }
        }
        
        for attempt in range(max_retries):
            # Ensure minimum delay between requests
            time_since_last = time.time() - self.last_request_time
            if time_since_last < self.min_delay:
                sleep_time = self.min_delay - time_since_last
                time.sleep(sleep_time)
            
            try:
                self.last_request_time = time.time()
                response = requests.post(url, headers=headers, json=payload, timeout=60)
                
                # Handle rate limiting with exponential backoff
                if response.status_code == 429:
                    # Much longer waits for 429 errors
                    wait_time = min((2 ** attempt) * 10, 300)  # 10, 20, 40, 80, 160, max 300s
                    print(f"  ⚠️  Rate limit hit. Waiting {wait_time} seconds... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                
                response.raise_for_status()
                result = response.json()
                
                if 'candidates' in result and len(result['candidates']) > 0:
                    return result['candidates'][0]['content']['parts'][0]['text']
                else:
                    print(f"  ✗ Gemini API returned no candidates")
                    return None
                    
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429 and attempt < max_retries - 1:
                    wait_time = min((2 ** attempt) * 10, 300)
                    print(f"  ⚠️  Rate limit (429). Waiting {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"  ✗ Gemini API error: {str(e)}")
                    if hasattr(e, 'response') and e.response is not None:
                        print(f"     Response: {e.response.text[:200]}")
                    return None
            except Exception as e:
                print(f"  ✗ Gemini API error: {str(e)}")
                return None
        
        print(f"  ✗ Max retries ({max_retries}) exceeded. Rate limit may require longer wait.")
        return None
    
    def generate_personalized_email(self, professor: ProfessorInfo) -> Optional[str]:
        """Generates a personalized email for the professor"""
        
        if not professor.email:
            print(f"  Skipping {professor.name} - no email found")
            return None
        
        # First API call: Generate the email content
        generation_prompt = f"""You are helping a biomedical engineering student write a personalized research internship application email.

STUDENT PROFILE:
{RESUME_SUMMARY}

PROFESSOR INFORMATION:
- Name: {professor.name}
- University: {professor.university}
- Department: {professor.department if professor.department else 'Not specified'}
- Research Areas: {', '.join(professor.research_areas) if professor.research_areas else 'Not specified'}

PROFESSOR'S RESEARCH DETAILS:
{professor.full_profile_text[:2000]}

INTERNSHIP DETAILS:
- Period: January 2026 to April 2026
- Student is currently at IIT Hyderabad, India
- Student did a REMOTE research internship at INSA Lyon (Jan-May 2025)
- Student will have completed the INSA Lyon internship before this new internship

REQUIRED EMAIL STRUCTURE (FOLLOW EXACTLY):
1. Opening: "Dear Prof. [Last Name], I hope this email finds you well."
2. Introduction: Introduce yourself - name, institution (IIT Hyderabad), degree (double major in Biomedical Engineering and Physics, minor in Economics), year (third year, 5th semester), academic standing (top 5% of class, CGPA: 9)
3. Express interest: State you're writing to express interest in joining their research group for a semester-long internship during January-April 2026
4. Research alignment: Explain what specifically interests you about the professor's research. Reference their specific research areas, methodologies, or contributions. Show you understand their work.
5. Your relevant background: Connect your background and projects to their research. Mention relevant coursework, projects (like 92% Dice coefficient brain tumor segmentation, INSA Lyon medical image annotation work), and technical skills (deep learning, PyTorch, computer vision, etc.)
6. Closing: Express gratitude for consideration, mention CV is attached, offer to provide additional information
7. Sign off: "Best regards, Parth Dawar"

CRITICAL REQUIREMENTS:
- Subject line MUST be exactly: "Application for Research Internship"
- Follow the structure above - introduce yourself FIRST, then express interest, then show your relevant work
- Be specific about the professor's research - no generic statements
- Keep it concise (250-350 words)
- Professional but warm tone
- Make clear you're currently at IIT Hyderabad and did a REMOTE internship at INSA Lyon
- Use proper paragraph breaks with blank lines between paragraphs
- No excessive spacing or line breaks

DO NOT use placeholder text or generic statements. Make every sentence specific and meaningful.

Format the email EXACTLY as:
Subject: Application for Research Internship

Dear Prof. [Last Name],

I hope this email finds you well.

[Introduction paragraph - single paragraph, no line breaks within]

[Interest in research paragraph - single paragraph, no line breaks within]

[Your background and skills paragraph - single paragraph, no line breaks within]

[Closing paragraph - single paragraph, no line breaks within]

Best regards,
Parth Dawar"""

        print(f"  Generating email for {professor.name}...")
        email_draft = self.call_gemini(generation_prompt)
        
        if not email_draft:
            return None
        
        # Second API call: Humanize the email
        humanization_prompt = f"""You are a writing editor specializing in making formal emails sound more natural and human.

Below is an email draft for a research internship application. Your task is to:

1. Make it sound more natural and conversational (while keeping it professional)
2. Remove any overly formal or robotic language
3. Ensure it flows smoothly
4. Keep all specific technical details and facts
5. Make sure it sounds like a genuine, enthusiastic student wrote it
6. Maintain the same structure and length
7. IMPORTANT: Keep proper paragraph formatting with blank lines between paragraphs
8. Each paragraph should be a single block of text without internal line breaks

ORIGINAL EMAIL:
{email_draft}

Return ONLY the improved email with proper formatting:
- Subject line first
- Blank line
- Dear Prof. [Name],
- Blank line  
- Each paragraph separated by a blank line
- No extra spacing within paragraphs
- Signature at the end

Do not add explanations or additional text."""

        print(f"  Humanizing email...")
        final_email = self.call_gemini(humanization_prompt)
        
        return final_email if final_email else email_draft


class GmailService:
    """Handles Gmail API authentication and draft creation"""
    
    SCOPES = ['https://www.googleapis.com/auth/gmail.compose']
    
    def __init__(self, resume_path: str):
        self.service = None
        self.resume_path = resume_path
        self.authenticated = False
        
        if not GMAIL_AVAILABLE:
            print("\n⚠️  Gmail functionality disabled - libraries not installed")
            return
        
        try:
            self.authenticate()
        except Exception as e:
            print(f"\n⚠️  Gmail authentication failed: {str(e)}")
            print("   Emails will be saved to files only, not drafted in Gmail")
    
    def authenticate(self):
        """Authenticates with Gmail API using OAuth2"""
        creds = None
        
        # Token file stores the user's access and refresh tokens
        if os.path.exists('token.pickle'):
            with open('token.pickle', 'rb') as token:
                creds = pickle.load(token)
        
        # If there are no valid credentials, let the user log in
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not os.path.exists('credentials.json'):
                    print("\n" + "="*80)
                    print("GMAIL API SETUP REQUIRED")
                    print("="*80)
                    print("\nTo create Gmail drafts, you need to set up Google API credentials:")
                    print("\n1. Go to: https://console.cloud.google.com/")
                    print("2. Create a new project (or select existing)")
                    print("3. Enable Gmail API:")
                    print("   - Go to 'APIs & Services' > 'Enable APIs and Services'")
                    print("   - Search for 'Gmail API' and enable it")
                    print("4. Create OAuth 2.0 credentials:")
                    print("   - Go to 'APIs & Services' > 'Credentials'")
                    print("   - Click 'Create Credentials' > 'OAuth client ID'")
                    print("   - Application type: 'Desktop app'")
                    print("   - Download the JSON file")
                    print("5. Save the downloaded JSON as 'credentials.json' in this directory")
                    print("\n📝 For detailed instructions, visit:")
                    print("   https://developers.google.com/gmail/api/quickstart/python")
                    print("\n⚠️  Continuing without Gmail integration...")
                    print("   Emails will be saved to files only")
                    print("="*80 + "\n")
                    return
                
                flow = InstalledAppFlow.from_client_secrets_file('credentials.json', self.SCOPES)
                creds = flow.run_local_server(port=0)
            
            # Save credentials for next run
            with open('token.pickle', 'wb') as token:
                pickle.dump(creds, token)
        
        self.service = build('gmail', 'v1', credentials=creds)
        self.authenticated = True
        print("✓ Gmail API authenticated successfully")
    
    def create_message_with_attachment(self, to: str, subject: str, body: str) -> dict:
        """Creates a message with resume attachment"""
        message = MIMEMultipart()
        message['to'] = to
        message['subject'] = subject
        
        # Convert plain text body to HTML with proper paragraph spacing
        paragraphs = body.split('\n\n')
        html_paragraphs = []
        
        for para in paragraphs:
            para = para.strip()
            if para:
                # Replace single newlines with <br> tags for line breaks within paragraphs
                para = para.replace('\n', '<br>')
                html_paragraphs.append(f'<p style="margin: 0 0 1em 0;">{para}</p>')
        
        html_body = f"""<html>
<body style="font-family: Arial, sans-serif; font-size: 14px; line-height: 1.6; color: #333;">
{''.join(html_paragraphs)}
</body>
</html>"""
        
        # Attach both plain text and HTML versions
        message.attach(MIMEText(body, 'plain'))
        message.attach(MIMEText(html_body, 'html'))
        
        # Attach resume
        if os.path.exists(self.resume_path):
            content_type, encoding = mimetypes.guess_type(self.resume_path)
            
            if content_type is None or encoding is not None:
                content_type = 'application/octet-stream'
            
            main_type, sub_type = content_type.split('/', 1)
            
            with open(self.resume_path, 'rb') as fp:
                attachment = MIMEBase(main_type, sub_type)
                attachment.set_payload(fp.read())
            
            encoders.encode_base64(attachment)
            filename = os.path.basename(self.resume_path)
            attachment.add_header('Content-Disposition', 'attachment', filename=filename)
            message.attach(attachment)
        else:
            print(f"  ⚠️  Resume not found at {self.resume_path}")
        
        raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode('utf-8')
        return {'raw': raw_message}
    
    def create_draft(self, to: str, subject: str, body: str) -> Optional[str]:
        """Creates a draft email in Gmail"""
        if not self.authenticated or not self.service:
            return None
        
        try:
            message = self.create_message_with_attachment(to, subject, body)
            draft = self.service.users().drafts().create(
                userId='me',
                body={'message': message}
            ).execute()
            
            draft_id = draft['id']
            print(f"  ✓ Gmail draft created (ID: {draft_id[:8]}...)")
            return draft_id
            
        except Exception as e:
            print(f"  ✗ Failed to create Gmail draft: {str(e)}")
            return None


class OutreachAutomation:
    """Main orchestration class"""
    
    def __init__(self, csv_path: str, api_key: str, resume_path: str):
        self.csv_path = csv_path
        self.resume_path = resume_path
        self.scraper = ProfessorScraper()
        self.email_generator = EmailGenerator(api_key)
        self.gmail_service = GmailService(resume_path) if GMAIL_AVAILABLE else None
        self.results = []
    
    def read_csv(self) -> List[Tuple[str, str]]:
        """Reads the CSV file and returns list of (university, url) tuples"""
        professors = []
        current_university = None
        
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                if not line.startswith('http'):
                    current_university = line
                else:
                    if current_university:
                        professors.append((current_university, line))
        
        return professors
    
    def process_professor(self, university: str, profile_url: str) -> Dict:
        """Processes a single professor"""
        # Scrape information
        professor = self.scraper.get_professor_info(profile_url, university)
        
        # Generate email if we have an email address
        email_content = None
        draft_id = None
        
        if professor.email:
            email_content = self.email_generator.generate_personalized_email(professor)
            
            # Save raw AI output for debugging (before any processing)
            if email_content:
                debug_dir = 'debug_emails'
                os.makedirs(debug_dir, exist_ok=True)
                debug_file = f"{debug_dir}/{professor.name.replace(' ', '_')}_RAW.txt"
                with open(debug_file, 'w', encoding='utf-8') as f:
                    f.write("=== RAW AI OUTPUT (BEFORE PROCESSING) ===\n\n")
                    f.write(email_content)
                    f.write("\n\n=== END RAW OUTPUT ===")
            
            # Create Gmail draft if authentication successful
            if email_content and self.gmail_service and self.gmail_service.authenticated:
                # First, check if AI generated duplicate content and clean it
                # Split by "Dear Prof." or "Dear Dr." to find duplicates
                import re
                
                # Find all occurrences of greeting
                greeting_pattern = r'Dear (?:Prof\.|Dr\.) \w+'
                greetings = list(re.finditer(greeting_pattern, email_content))
                
                # If multiple greetings found, keep only content up to second greeting
                if len(greetings) > 1:
                    # Keep only content from start to second greeting (exclusive)
                    email_content = email_content[:greetings[1].start()].strip()
                
                # Parse email to extract subject and body
                lines = email_content.split('\n')
                subject = "Application for Research Internship"  # Default subject
                body_lines = []
                
                in_body = False
                for i, line in enumerate(lines):
                    if line.startswith('Subject:'):
                        subject = line.replace('Subject:', '').strip()
                    elif line.startswith('Dear Prof.') or line.startswith('Dear Dr.'):
                        # Start of email body
                        in_body = True
                        body_lines.append(line)
                    elif in_body:
                        body_lines.append(line)
                
                body = '\n'.join(body_lines).strip()
                
                # Additional check: if body is longer than 600 words, likely duplicated
                word_count = len(body.split())
                if word_count > 600:
                    # Split in half and check if halves are similar (duplication)
                    half_point = len(body) // 2
                    first_half = body[:half_point]
                    second_half = body[half_point:]
                    
                    # If second half starts with greeting again, it's a duplicate
                    if 'Dear Prof.' in second_half or 'Dear Dr.' in second_half:
                        # Find where the duplication starts
                        dup_start = body.find('Dear Prof.', 50)  # Skip first greeting
                        if dup_start == -1:
                            dup_start = body.find('Dear Dr.', 50)
                        if dup_start > 0:
                            body = body[:dup_start].strip()
                
                # Clean up excessive newlines (more than 2 consecutive)
                body = re.sub(r'\n{3,}', '\n\n', body)
                
                # Ensure proper paragraph spacing
                body = body.replace('\n\n', '\n\n')  # Normalize double newlines
                
                # Add signature if not present
                if 'Parth Dawar' not in body[-200:]:
                    body += "\n\nBest regards,\nParth Dawar\nB.Tech in Biomedical Engineering\nIIT Hyderabad\nbm23btech11018@iith.ac.in\n+919814433323"
                
                # --- MODIFICATION START: Decode email address ---
                # Decode URL-encoded characters (like %40 for @)
                decoded_email = unquote(professor.email)
                # --- MODIFICATION END ---
                
                draft_id = self.gmail_service.create_draft(
                    to=decoded_email, # Pass the decoded email
                    subject=subject,
                    body=body
                )
            
            time.sleep(2)  # Rate limiting - wait between API calls
        
        result = {
            'name': professor.name,
            'university': professor.university,
            'email': professor.email,
            'profile_url': professor.profile_url,
            'research_areas': professor.research_areas,
            'department': professor.department,
            'email_generated': email_content,
            'gmail_draft_id': draft_id,
            'status': 'SUCCESS' if email_content else ('NO_EMAIL' if not professor.email else 'GENERATION_FAILED')
        }
        
        return result
    
    def run(self, limit: Optional[int] = None, resume: bool = False):
        """Runs the complete automation process"""
        print("="*80)
        print("PROFESSOR OUTREACH AUTOMATION SYSTEM")
        print("="*80)
        
        professors = self.read_csv()
        print(f"\nFound {len(professors)} professors to process")
        
        # Check for existing results to resume (only if resume=True)
        already_processed = set()
        if resume and os.path.exists('outreach_results.json'):
            try:
                with open('outreach_results.json', 'r') as f:
                    existing_results = json.load(f)
                    already_processed = {r['profile_url'] for r in existing_results if r.get('status') == 'SUCCESS'}
                    if already_processed:
                        print(f"Found {len(already_processed)} already processed professors")
                        print("Resuming from where you left off...")
                        self.results = existing_results
            except:
                pass
        else:
            # Starting fresh - clear old results
            self.results = []
            if os.path.exists('outreach_results.json'):
                print("Starting fresh - will process all professors from beginning")
        
        if limit:
            professors = professors[:limit]
            print(f"Processing first {limit} professors (limit set)")
        
        print("\nStarting processing...\n")
        
        processed_count = 0
        for i, (university, profile_url) in enumerate(professors, 1):
            # Skip if already successfully processed
            if profile_url in already_processed:
                print(f"\n[{i}/{len(professors)}] ⏩ Skipping (already processed)")
                processed_count += 1
                continue
            
            print(f"\n[{i}/{len(professors)}] Processing...")
            try:
                result = self.process_professor(university, profile_url)
                
                # Update results (replace if exists, add if new)
                existing_idx = next((idx for idx, r in enumerate(self.results) if r.get('profile_url') == profile_url), None)
                if existing_idx is not None:
                    self.results[existing_idx] = result
                else:
                    self.results.append(result)
                
                print(f"  Status: {result['status']}")
                processed_count += 1
                
                # Save after each successful processing (in case of interruption)
                if processed_count % 5 == 0:  # Save every 5 professors
                    self.save_results()
                    print(f"  💾 Progress saved ({processed_count} processed)")
                
            except KeyboardInterrupt:
                print("\n\n⚠️  Interrupted by user. Saving progress...")
                self.save_results()
                print("✓ Progress saved. Run again to resume from here.")
                return
            except Exception as e:
                print(f"  ERROR: {str(e)}")
                self.results.append({
                    'name': 'ERROR',
                    'university': university,
                    'profile_url': profile_url,
                    'status': 'ERROR',
                    'error': str(e)
                })
            
            time.sleep(5)  # Increased delay to respect Gemini rate limits
        
        self.save_results()
        self.print_summary()
    
    def save_results(self):
        """Saves results to files"""
        # Save detailed JSON
        with open('outreach_results.json', 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        # Save emails to individual files
        os.makedirs('generated_emails', exist_ok=True)
        
        successful = 0
        for result in self.results:
            if result.get('email_generated'):
                filename = f"generated_emails/{result['name'].replace(' ', '_')}.txt"
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(f"To: {result['email']}\n")
                    f.write(f"Professor: {result['name']}\n")
                    f.write(f"University: {result['university']}\n")
                    f.write(f"Profile: {result['profile_url']}\n")
                    f.write("\n" + "="*80 + "\n\n")
                    f.write(result['email_generated'])
                successful += 1
        
        print(f"\n✓ Saved {successful} emails to 'generated_emails/' directory")
        print(f"✓ Saved detailed results to 'outreach_results.json'")
    
    def print_summary(self):
        """Prints a summary of the automation run"""
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        
        total = len(self.results)
        successful = sum(1 for r in self.results if r['status'] == 'SUCCESS')
        no_email = sum(1 for r in self.results if r['status'] == 'NO_EMAIL')
        failed = sum(1 for r in self.results if r['status'] in ['GENERATION_FAILED', 'ERROR'])
        gmail_drafts = sum(1 for r in self.results if r.get('gmail_draft_id'))
        
        print(f"\nTotal Professors: {total}")
        print(f"✓ Successfully Generated: {successful}")
        print(f"✗ No Email Found: {no_email}")
        print(f"✗ Failed: {failed}")
        print(f"\nSuccess Rate: {(successful/total*100):.1f}%")
        
        if self.gmail_service and self.gmail_service.authenticated:
            print(f"\n📧 Gmail Drafts Created: {gmail_drafts}")
            if gmail_drafts > 0:
                print("   View drafts at: https://mail.google.com/mail/#drafts")
        
        if successful > 0:
            print(f"\n🎉 {successful} personalized emails ready!")
            print("Check the 'generated_emails/' folder for individual email files.")
            if gmail_drafts > 0:
                print(f"✓ {gmail_drafts} emails also drafted in your Gmail account with resume attached!")


def main():
    """Main entry point"""
    import sys
    
    # Check for API key
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        print("ERROR: GEMINI_API_KEY environment variable not set")
        print("\nPlease set your Gemini API key:")
        print("  export GEMINI_API_KEY='your-api-key-here'")
        print("\nGet your API key at: https://makersuite.google.com/app/apikey")
        print("(It's 100% FREE!)")
        sys.exit(1)
    
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python3 professor_outreach_automation.py CSV_FILE [RESUME_FILE] [LIMIT]")
        print("\nExamples:")
        print("  python3 professor_outreach_automation.py foriegn_unis.csv Resume.pdf 3")
        print("  python3 professor_outreach_automation.py foriegn_unis.csv Resume.pdf")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    
    if not os.path.exists(csv_path):
        print(f"ERROR: CSV file not found: {csv_path}")
        sys.exit(1)
    
    # Determine resume path and limit
    resume_path = 'Resume.pdf'  # Default
    limit = None
    
    if len(sys.argv) >= 3:
        # Check if arg is a number (limit) or a file (resume)
        arg2 = sys.argv[2]
        if arg2.isdigit():
            # It's a limit, use default resume
            limit = int(arg2)
        elif os.path.exists(arg2):
            # It's a resume file
            resume_path = arg2
            # Check for limit in 4th position
            if len(sys.argv) >= 4 and sys.argv[3].isdigit():
                limit = int(sys.argv[3])
        else:
            print(f"WARNING: '{arg2}' is neither a valid file nor a number")
            print(f"Using default Resume.pdf and treating '{arg2}' as limit if numeric")
            if arg2.isdigit():
                limit = int(arg2)
    
    if not os.path.exists(resume_path):
        print(f"WARNING: Resume file not found: {resume_path}")
        print("Gmail drafts will be created without attachment.")
        print("Please ensure Resume.pdf is in the current directory.\n")
    
    print("\n" + "="*80)
    print("GMAIL INTEGRATION NOTES")
    print("="*80)
    print("\nThis script will:")
    print("1. Generate personalized emails using AI")
    print("2. Save them as text files in 'generated_emails/' folder")
    print("3. Create drafts in your Gmail account with resume attached")
    print("\nFor Gmail integration, you'll need to:")
    print("- Have 'credentials.json' from Google Cloud Console")
    print("- Authenticate once (browser will open)")
    print("- Credentials will be saved for future runs")
    print("\nIf Gmail setup is skipped, emails will only be saved to files.")
    print("="*80 + "\n")
    
    # Run automation
    automation = OutreachAutomation(csv_path, api_key, resume_path)
    automation.run(limit=limit)
    
    print("\n✓ Automation complete!")
    print("\nNext steps:")
    print("1. Check Gmail drafts: https://mail.google.com/mail/#drafts")
    print("2. Review and edit drafts as needed")
    print("3. Send emails to professors")
    print("4. Track responses")


if __name__ == "__main__":
    main()