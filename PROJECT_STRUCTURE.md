# Smart Attendance System - Complete Project Documentation

## 📌 Project Overview

**Smart Attendance System** is a comprehensive face recognition-based attendance tracking application built with Flask, OpenCV, and SQLite. It features dual portal architecture for both institutional administrators (gate entry management) and teachers (classroom attendance management).

**Author:** Manish Sharma  
**Version:** 2.0  
**License:** Custom License (see LICENSE file)  
**Technology Stack:** Python (Flask), SQLite, OpenCV, KNN Classifier, Cloudinary

---

## 🏗️ Complete Directory Structure

```
Smart Attendance System/
├── app.py                                    # Main Flask application (~6,000 lines)
├── db.sqlite3                                # SQLite database (attendance records)
├── attendance.db                             # Additional database file
├── haarcascade_frontalface_default.xml       # OpenCV face detection model
├── requirements.txt                          # Python dependencies (20 packages)
├── schema.sql                                # Database schema definition
├── README.md                                 # Project readme
├── .flask_secret                             # Session secret key
├── LICENSE                                   # Custom project license
│
├── __pycache__/                              # Python bytecode cache
├── .venv/                                    # Virtual environment
│
├── data/                                     # Face recognition data
│   ├── faces_data.pkl                        # Serialized face encodings (NumPy arrays)
│   └── names.pkl                             # Registered user names list
│
├── Attendance/                               # Daily attendance exports
│   └── Attendance_*.csv                      # Daily CSV records with timestamp
│
├── static/                                   # Frontend static assets
│   ├── css/
│   │   ├── main.css                          # Core styling, navigation, color scheme
│   │   ├── capture.css                       # Face capture UI styling
│   │   ├── recognize.css                     # Recognition interface styling
│   │   ├── attendance.css                    # Attendance table styling
│   │   └── animations.css                    # CSS animations and transitions
│   │
│   └── js/
│       ├── main.js                           # Global utilities, API helpers
│       ├── capture.js                        # Face capture implementation
│       ├── dashboard.js                      # Dashboard statistics and charts
│       └── attendance.js                     # Attendance table interactions
│
└── templates/                                # Jinja2 HTML templates
    ├── base.html                             # Base layout (navbar, footer, flash messages)
    ├── index.html                            # Home page
    ├── instructions.html                     # User instructions
    ├── error.html                            # Error page
    ├── result.html                           # Result display page
    ├── settings.html                         # Settings page
    ├── capture.html                          # Face capture interface
    ├── recognize.html                        # Face recognition interface
    ├── attendance.html                       # Attendance view/export
    ├── export_attendance.html                # Attendance export interface
    ├── manage_users.html                     # User management interface
    │
    ├── auth/                                 # Authentication templates
    │   ├── login.html                        # Teacher/staff PIN login
    │   ├── register.html                     # Teacher registration (disabled)
    │   ├── verify.html                       # Email OTP verification
    │   ├── forgot_password.html              # Password reset request
    │   ├── verify_password_otp.html          # Password OTP verification
    │   ├── reset_password.html               # New password entry
    │   ├── forgot_pin.html                   # PIN reset request
    │   ├── verify_pin_otp.html               # PIN OTP verification
    │   └── reset_pin.html                    # New PIN entry
    │
    ├── admin/                                # Admin portal templates
    │   ├── admin_login.html                  # Admin email/password login
    │   ├── admin_register.html               # Admin self-registration
    │   ├── admin_verify.html                 # Admin email verification
    │   ├── dashboard.html                    # Admin dashboard with stats
    │   ├── attendance.html                   # Admin attendance view
    │   └── profile.html                      # Admin profile management
    │
    └── teacher/                              # Teacher portal templates
        ├── dashboard.html                    # Teacher dashboard with stats
        ├── classes.html                      # Class list view
        ├── class_detail.html                 # Individual class details
        ├── attendance.html                   # Teacher attendance view
        ├── student_attendance.html           # Per-student attendance details
        └── profile.html                      # Teacher profile management
```

---

## 🗄️ Database Architecture

### Database Tables (10 tables)

#### 1. **teachers** - Teacher/Staff Accounts
```sql
Columns:
- id (PRIMARY KEY)
- username (UNIQUE) - Login username
- password_hash - SHA256 hashed password
- email (UNIQUE) - Email address
- full_name - Full name
- phone - Contact number
- department - Department/subject
- designation - Job title
- bio - Profile bio
- profile_image - Cloudinary image URL
- user_id (UNIQUE) - Auto-generated (TCH-XXXX, STF-XXXX)
- user_pin - 6-digit authentication PIN
- created_at - Registration timestamp
```

#### 2. **admins** - Administrator Accounts
```sql
Columns:
- id (PRIMARY KEY)
- username (UNIQUE)
- password_hash - SHA256 hashed password
- email (UNIQUE)
- full_name
- college_name - Institution name
- phone
- designation
- bio
- profile_image
- user_id (UNIQUE) - Auto-generated (ADM-XXXX)
- user_pin - 6-digit authentication PIN
- institution_type - School/College
- created_at
```

#### 3. **users** - Registered Students/Staff
```sql
Columns:
- id (PRIMARY KEY)
- username (UNIQUE)
- name - Full name
- email (UNIQUE)
- user_id (UNIQUE) - Auto-generated (STU-XXXX, STF-XXXX)
- department
- phone
- role - Student/Staff
- notes - Additional information
- created_at
- registered_by - Teacher ID who registered the user
- registered_by_admin_id - Admin ID (if registered by admin)
- is_active - Active status flag (1=active, 0=deactivated)
```

#### 4. **classes** - Teacher-Created Classes
```sql
Columns:
- id (PRIMARY KEY)
- teacher_id (FOREIGN KEY → teachers.id)
- name - Class name
- description - Class description
- department - Associated department
- created_at
```

#### 5. **admin_classes** - Institution-Wide Classes
```sql
Columns:
- id (PRIMARY KEY)
- name - Class name
- institution_type - School/College
- department
- description
- created_at
```

#### 6. **class_students** - Student-Class Mapping
```sql
Columns:
- class_id (FOREIGN KEY → classes.id)
- student_name - Student name (matches users.name)
PRIMARY KEY: (class_id, student_name)
```

#### 7. **teacher_class_assignments** - Teacher-to-AdminClass Mapping
```sql
Columns:
- id (PRIMARY KEY)
- admin_class_id (FOREIGN KEY → admin_classes.id)
- teacher_id (FOREIGN KEY → teachers.id)
- assigned_at
```

#### 8. **attendance_records** - Attendance Logs
```sql
Columns:
- id (PRIMARY KEY)
- student_name - Student name
- class_id (FOREIGN KEY → classes.id, nullable for gate attendance)
- teacher_id (FOREIGN KEY → teachers.id, nullable)
- admin_id (FOREIGN KEY → admins.id, nullable)
- date - Attendance date (YYYY-MM-DD)
- time - Attendance time (HH:MM:SS)
- status - Present/Absent/Late
- attendance_type - 'gate' or 'class'
- notes - Additional notes
- created_at
```

#### 9. **email_verifications** - Pending Email Verifications
```sql
Columns:
- id (PRIMARY KEY)
- email
- username
- password_hash
- full_name
- otp_code - 5-digit verification code
- created_at
- expires_at - 10-minute expiry
- verified - Verification status (0/1)
- user_type - 'teacher' or 'admin'
```

#### 10. **password_resets** - Password/PIN Reset Tokens
```sql
Columns:
- id (PRIMARY KEY)
- teacher_id (nullable)
- admin_id (nullable)
- token - Reset token (legacy)
- created_at
- expires_at
- used - Used status (0/1)
- user_type - 'teacher' or 'admin'
```

---

## 🚀 Technology Stack

### Backend Technologies
| Technology | Version | Purpose |
|-----------|---------|---------|
| Python | 3.8+ | Core programming language |
| Flask | 3.1.2 | Web framework |
| SQLite | 3.x | Relational database |
| OpenCV | 4.12.0.88 | Computer vision (face detection) |
| Scikit-learn | 1.8.0 | Machine learning (KNN classifier) |
| NumPy | 2.2.6 | Numerical computing (face encoding arrays) |
| Jinja2 | 3.1.6 | Template engine |

### Frontend Technologies
| Technology | Purpose |
|-----------|---------|
| HTML5 | Markup structure |
| CSS3 | Styling, animations, responsive design |
| JavaScript (Vanilla) | Client-side logic, API calls |
| Bootstrap 4 | CSS framework |
| Font Awesome | Icon library |

### External Services
| Service | Purpose | Configuration |
|---------|---------|---------------|
| Cloudinary | Profile image storage | cloud_name: "dud3f00ay" |
| Gmail SMTP | Email delivery (OTP, credentials) | mpandat0052@gmail.com:587 |

### Python Dependencies (requirements.txt)
```
asgiref==3.11.0              # ASGI utilities
blinker==1.9.0               # Flask signals
click==8.3.1                 # CLI framework
cloudinary>=1.36.0           # Cloud media storage
colorama==0.4.6              # Terminal colors
Django==6.0                  # (Legacy, not used)
Flask==3.1.2                 # Web framework ⭐
itsdangerous==2.2.0          # Token security
Jinja2==3.1.6                # Template engine
joblib==1.5.2                # Parallel computing
MarkupSafe==3.0.3            # HTML escaping
numpy==2.2.6                 # Numerical computing ⭐
opencv-python==4.12.0.88     # Computer vision ⭐
pywin32==311                 # Windows COM (text-to-speech)
scikit-learn==1.8.0          # KNN classifier ⭐
scipy==1.16.3                # Scientific computing
sqlparse==0.5.4              # SQL parsing
threadpoolctl==3.6.0         # Thread management
tzdata==2025.2               # Timezone data
Werkzeug==3.1.4              # WSGI utilities
```

---

## 📡 Complete API Endpoints (50+ Routes)

### Authentication Routes (Teacher/Staff)
| Method | Endpoint | Purpose | Authentication |
|--------|----------|---------|----------------|
| GET/POST | `/auth/register` | Teacher registration | PUBLIC (disabled) |
| GET/POST | `/auth/verify` | Email OTP verification | PUBLIC |
| POST | `/auth/resend-otp` | Resend verification code | PUBLIC |
| GET/POST | `/auth/pin` | Teacher PIN login (User ID + PIN) | PUBLIC |
| GET/POST | `/auth/forgot-password` | Password reset request | PUBLIC |
| POST | `/auth/verify-password-otp` | Verify password reset OTP | PUBLIC |
| GET/POST | `/auth/reset-password` | Set new password | SESSION |
| GET/POST | `/auth/forgot-pin` | PIN reset request | PUBLIC |
| POST | `/auth/verify-pin-otp` | Verify PIN reset OTP | PUBLIC |
| GET/POST | `/auth/reset-pin` | Set new PIN | SESSION |
| POST | `/auth/logout` | Teacher logout | AUTHENTICATED |

### Admin Authentication Routes
| Method | Endpoint | Purpose | Authentication |
|--------|----------|---------|----------------|
| GET/POST | `/admin/login` | Admin email + password login | PUBLIC |
| GET/POST | `/admin/register` | Admin self-registration | PUBLIC |
| GET/POST | `/admin/verify` | Admin email verification | PUBLIC |
| POST | `/admin/resend-otp` | Resend admin verification code | PUBLIC |
| POST | `/admin/logout` | Admin logout | AUTHENTICATED |

### Face Capture & Recognition Routes
| Method | Endpoint | Purpose | Authentication |
|--------|----------|---------|----------------|
| GET | `/capture` | Face capture page | AUTHENTICATED |
| POST | `/api/upload_capture` | Web-based face capture (base64) | AUTHENTICATED |
| GET | `/recognize` | Face recognition page | AUTHENTICATED |
| POST | `/api/capture_frame` | Real-time face detection | AUTHENTICATED |
| POST | `/api/save_attendance` | Save recognized attendance | AUTHENTICATED |
| GET | `/api/get_recognized_users` | List all registered users | AUTHENTICATED |

### Attendance Management Routes
| Method | Endpoint | Purpose | Authentication |
|--------|----------|---------|----------------|
| GET | `/attendance` | View attendance with filters | AUTHENTICATED |
| GET/POST | `/export_attendance` | Export attendance (CSV/JSON) | AUTHENTICATED |
| GET | `/api/export_preview` | Preview export data | AUTHENTICATED |
| POST | `/teacher/export` | Teacher export attendance | TEACHER |
| POST | `/teacher/delete_record/<id>` | Delete attendance record | TEACHER |
| GET | `/teacher/export_today` | Export today's attendance | TEACHER |
| POST | `/teacher/attendance/filter` | Filter attendance | TEACHER |
| GET | `/admin/attendance` | Admin attendance view (defaults to today) | ADMIN |
| POST | `/admin/attendance/filter` | Filter admin attendance | ADMIN |

### User Management Routes
| Method | Endpoint | Purpose | Authentication |
|--------|----------|---------|----------------|
| GET | `/manage_users` | View registered users | AUTHENTICATED |
| POST | `/delete_user/<username>` | Delete user | AUTHENTICATED |
| POST | `/api/edit_user/<username>` | Edit user details | AUTHENTICATED |
| GET | `/api/admin/all_users` | Admin get all users | ADMIN |
| POST | `/api/admin/users/<id>/update` | Admin update user | ADMIN |
| POST | `/api/admin/users/<id>/toggle_active` | Enable/disable user | ADMIN |
| DELETE | `/api/admin/users/<id>/delete` | Delete user with face data cleanup | ADMIN |
| POST | `/api/admin/create_teacher` | Admin create teacher account | ADMIN |
| GET | `/api/next_user_id` | Generate next user ID | AUTHENTICATED |

### Class Management Routes
| Method | Endpoint | Purpose | Authentication |
|--------|----------|---------|----------------|
| GET | `/teacher/classes` | List teacher's classes | TEACHER |
| POST | `/teacher/classes` | Create new class | TEACHER |
| GET | `/teacher/class/<id>` | View class details | TEACHER |
| POST | `/teacher/edit_class/<id>` | Edit class info | TEACHER |
| POST | `/teacher/class/<id>/student_attendance` | View class attendance | TEACHER |
| POST | `/filter_class_student_attendance/<id>` | Filter class attendance | TEACHER |
| POST | `/teacher/add_student_to_class/<id>` | Add student to class | TEACHER |
| POST | `/teacher/remove_student_from_class/<id>` | Remove student from class | TEACHER |
| POST | `/teacher/delete_class/<id>` | Delete class | TEACHER |
| GET | `/api/classes/for_capture` | Get classes for capture page | AUTHENTICATED |
| GET | `/api/admin/classes` | Admin get all classes | ADMIN |
| POST | `/api/admin/classes` | Admin create class | ADMIN |
| DELETE | `/api/admin/classes/<id>` | Admin delete class | ADMIN |
| POST | `/api/admin/classes/<id>/assign` | Assign teacher to class | ADMIN |
| DELETE | `/api/admin/classes/<id>/unassign/<tid>` | Unassign teacher | ADMIN |
| GET | `/api/admin/teachers` | List all teachers for assignment | ADMIN |

### Dashboard & Profile Routes
| Method | Endpoint | Purpose | Authentication |
|--------|----------|---------|----------------|
| GET | `/` | Home page | PUBLIC |
| GET | `/instructions` | Instructions page | PUBLIC |
| GET | `/teacher/dashboard` | Teacher dashboard with stats | TEACHER |
| GET | `/teacher/profile` | View/edit teacher profile | TEACHER |
| POST | `/teacher/profile` | Update teacher profile | TEACHER |
| POST | `/teacher/profile/remove-image` | Remove profile image | TEACHER |
| GET | `/api/teacher/stats` | Get teacher dashboard stats | TEACHER |
| POST | `/api/teacher/contact_admin` | Send help request email | TEACHER |
| GET | `/admin/dashboard` | Admin dashboard with stats | ADMIN |
| GET | `/admin/profile` | View/edit admin profile | ADMIN |
| POST | `/admin/profile` | Update admin profile | ADMIN |
| POST | `/admin/profile/remove-image` | Remove admin profile image | ADMIN |
| GET | `/api/admin/stats` | Get admin dashboard stats | ADMIN |
| GET | `/admin/settings` | Settings page | ADMIN |
| POST | `/admin/settings` | Save settings | ADMIN |

### Utility Routes
| Method | Endpoint | Purpose | Authentication |
|--------|----------|---------|----------------|
| GET | `/api/departments` | Get departments list | AUTHENTICATED |
| GET | `/api/options` | Get merged options (departments/roles) | AUTHENTICATED |
| GET | `/test_attendance` | Test attendance creation | PUBLIC |

---

## 🎯 Feature Set

### 1. Admin Portal Features
✅ **Gate Entry Management**
- Real-time face recognition at college/school gate
- All students/staff attendance recording
- Gate attendance marked with admin ID

✅ **Teacher Account Management**
- Create teacher/staff accounts with instant credentials
- Auto-generate User ID (TCH-XXXX, STF-XXXX)
- Email credentials to newly created teachers
- View all teachers

✅ **Global Class Management**
- Create institution-wide classes (School/College)
- Assign multiple teachers to classes
- Unassign teachers from classes
- Delete classes

✅ **User Management**
- View all registered students/staff
- Edit user details (name, email, department, phone)
- Activate/deactivate users (soft delete)
- Delete users with automatic face data cleanup
- Search and filter users

✅ **Attendance Oversight**
- View ALL attendance records (gate + class)
- Default view: Today's records only
- Filter by date range, type (gate/class), department, status
- Export attendance (CSV/JSON)
- View attendance trends and statistics

✅ **Profile Management**
- Update admin profile (name, email, phone, designation)
- Upload profile image (Cloudinary)
- Remove profile image
- Change password
- Change PIN

✅ **Dashboard Analytics**
- Total registered users count
- Total attendance count
- Weekly attendance trends
- Department-wise distribution
- Recent attendance activity

✅ **Settings**
- Institution configuration
- Email settings
- System preferences

---

### 2. Teacher Portal Features
✅ **Class Management**
- Create multiple classes (subject-based or section-based)
- Edit class details (name, description, department)
- View class list with student counts
- Delete classes (cascade delete students)

✅ **Student Registration**
- Face capture interface (50+ face samples per person)
- Web-based capture (canvas + webcam)
- Auto-generate Student ID (STU-XXXX)
- Add students to specific classes
- Remove students from classes

✅ **Class Attendance**
- Real-time face recognition for class attendance
- Class-specific attendance recording
- Duplicate prevention (1 entry per day per class)
- Confidence-based recognition (KNN voting)
- Manual attendance fallback

✅ **Attendance Tracking**
- View attendance by class/date/student
- Filter attendance records
- Per-student attendance rate calculation
- Per-class attendance rate
- Export today's attendance (CSV)
- Delete incorrect attendance records

✅ **Student Performance**
- View individual student attendance history
- Attendance percentage calculation
- Date range filtering for student attendance
- Visual indicators (Present/Absent/Late)

✅ **Profile Management**
- Update teacher profile (name, email, phone, department)
- Upload profile image (Cloudinary)
- Remove profile image
- Change password
- Change PIN

✅ **Dashboard Analytics**
- Total classes count
- Total students count
- Today's attendance count
- Recent attendance activity
- Class-wise statistics

✅ **Help & Support**
- Contact admin via email (help request)
- Instructions page
- Profile bio/designation display

---

### 3. Face Recognition System
✅ **Face Detection**
- Haar Cascade Frontal Face Detector (OpenCV)
- Multiple face detection in single frame
- Real-time webcam feed processing
- DirectShow backend for Windows webcam

✅ **Face Recognition Algorithm**
- K-Nearest Neighbors (KNN) classifier
- 50+ face samples per person (configurable)
- Confidence-based recognition (voting from k neighbors)
- Pickle serialization for face data storage
- NumPy array storage for face encodings

✅ **Capture Modes**
- **Web Capture:** Canvas-based image upload (base64 encoding)
- **Webcam Capture:** Real-time OpenCV capture
- Multiple images per person for training

✅ **Recognition Features**
- Real-time recognition from webcam feed
- Confidence threshold filtering
- Unknown face handling
- Manual attendance fallback
- Support for multiple faces in frame

✅ **Data Management**
- Face data stored in `data/faces_data.pkl` (NumPy arrays)
- Names stored in `data/names.pkl` (Python list)
- Automatic data update on new registration
- Face data cleanup on user deletion

---

### 4. Attendance System
✅ **Dual Attendance Types**
- **Gate Attendance:** Institution entry/exit (admin-managed)
- **Class Attendance:** Classroom attendance (teacher-managed)

✅ **Attendance Recording**
- Real-time face recognition
- Manual entry (fallback)
- Status tracking (Present/Absent/Late)
- Timestamp recording (date + time)
- Notes/remarks field

✅ **Duplicate Prevention**
- 1 gate attendance per day per student
- 1 class attendance per day per class per student
- Database-level duplicate checks
- "Already marked" notifications

✅ **Attendance Export**
- Daily CSV export (`Attendance_DD-MM-YYYY.csv`)
- Custom date range export
- JSON export support
- Export preview before download
- Columns: Student Name, Date, Time, Status, Type, Class, Notes

✅ **Attendance Analytics**
- Daily attendance count
- Weekly/monthly trends
- Department-wise statistics
- Class-wise statistics
- Student-wise attendance rate
- Graphical visualizations

✅ **Attendance Filtering**
- Filter by date range
- Filter by attendance type (gate/class)
- Filter by department
- Filter by status (Present/Absent/Late)
- Filter by student name
- Filter by class

---

### 5. Authentication & Security
✅ **Multiple Authentication Methods**
- Email + Password (admin, teacher)
- User ID + PIN (admin, teacher)
- Dual support for flexible login

✅ **Email Verification**
- 5-digit OTP codes
- 10-minute expiry
- Resend OTP functionality
- HTML email templates
- Verification required before login

✅ **Password Reset**
- OTP-based password reset (5-digit code)
- Email delivery
- 10-minute OTP expiry
- Session-based reset flow
- 30-second resend cooldown

✅ **PIN Reset**
- OTP-based PIN reset (5-digit code)
- Email delivery
- 10-minute OTP expiry
- Session-based reset flow
- 4-6 digit numeric PIN validation

✅ **Security Features**
- SHA256 password hashing
- Session-based authentication
- 30-day persistent sessions ("Remember Me")
- Secure token generation (secrets module)
- CSRF protection (Flask default)
- SQL injection prevention (parameterized queries)

✅ **Role-Based Access Control**
- Decorator-based authentication (`@teacher_login_required`, `@admin_login_required`)
- Role-specific route protection
- Session-based role tracking
- Automatic route redirection based on role

---

## 🔧 Installation & Setup

### Prerequisites
```bash
Python 3.8 or higher
pip (Python package manager)
Webcam (for face capture/recognition)
SMTP email account (Gmail recommended)
Cloudinary account (for profile images)
Windows OS (for pywin32 text-to-speech)
```

### Installation Steps

#### 1. Clone Repository
```bash
git clone <repository-url>
cd "Smart Attendence System"
```

#### 2. Create Virtual Environment
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac
```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 4. Database Setup
```bash
# Database will be created automatically on first run
# Or manually initialize:
python -c "from app import init_db; init_db()"
```

#### 5. Configuration
Update the following in `app.py` (lines 50-70):
```python
# Email Configuration
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USERNAME = "your-email@gmail.com"
SMTP_PASSWORD = "your-app-password"  # Use App Password, not regular password

# Cloudinary Configuration
cloudinary.config(
    cloud_name = "your-cloud-name",
    api_key = "your-api-key",
    api_secret = "your-api-secret"
)
```

**⚠️ Security Best Practice:** Use environment variables instead of hardcoding credentials:
```python
import os
SMTP_USERNAME = os.getenv("SMTP_USERNAME")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
cloudinary.config(
    cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key = os.getenv("CLOUDINARY_API_KEY"),
    api_secret = os.getenv("CLOUDINARY_API_SECRET")
)
```

#### 6. Run Application
```bash
python app.py
```
Application will start on `http://127.0.0.1:5000`

---

## 👥 User Roles & Permissions

### Admin Role
**Access:** Admin Portal (`/admin/*`)
**Capabilities:**
- Gate attendance management (all students)
- Create/manage teacher accounts
- Manage institution-wide classes
- Assign teachers to classes
- View all users (students, teachers, staff)
- Activate/deactivate users
- View all attendance records (gate + class)
- Export attendance
- Dashboard analytics
- Profile management
- System settings

**User ID Format:** `ADM-XXXX` (auto-generated, 4-digit number)
**Authentication:** Email + Password OR User ID + PIN

---

### Teacher Role
**Access:** Teacher Portal (`/teacher/*`)
**Capabilities:**
- Create/manage own classes
- Register students (face capture)
- Add/remove students from classes
- Class-based attendance (via face recognition)
- View class attendance
- View student attendance history
- Export attendance (CSV)
- Dashboard analytics
- Profile management
- Contact admin (help request)

**User ID Format:** `TCH-XXXX` (teachers), `STF-XXXX` (staff)
**Authentication:** Email + Password OR User ID + PIN

---

### Student Role
**Access:** None (students don't log in)
**Representation:**
- Registered in `users` table
- Face data stored in `data/faces_data.pkl`
- Enrolled in classes via `class_students` table
- Attendance tracked in `attendance_records` table

**User ID Format:** `STU-XXXX` (auto-generated, 4-digit number)
**No Login:** Students are recognized via face recognition only

---

## 🎨 UI/UX Design

### Color Scheme
| Role | Primary Color | Gradient | Usage |
|------|--------------|----------|-------|
| **Admin** | Pink (#f5576c) | Pink-to-Red | Login, dashboard, buttons, headers |
| **Teacher** | Purple (#667eea) | Purple-to-Indigo | Login, dashboard, buttons, headers |
| **Success** | Green (#48bb78) | Green gradient | Forms, success messages |
| **Accent** | Teal (#38b2ac) | Teal gradient | Highlights, links |

### Responsive Design
- Mobile-first approach
- Bootstrap 4 grid system
- Responsive tables with horizontal scroll
- Hamburger menu for mobile navigation
- Touch-friendly buttons and inputs

### UI Components
- **Cards:** Dashboard stats, class cards, user cards
- **Tables:** Attendance tables, user tables, class tables
- **Forms:** Login, registration, profile update, settings
- **Modals:** Confirmation dialogs, user edit, class edit
- **Toast Notifications:** Flash messages, success/error alerts
- **Charts:** Attendance trends (via Chart.js or similar)
- **Webcam Feed:** Live video canvas with overlay

---

## 📊 Data Flow Architecture

### 1. User Registration Flow (Teacher/Admin)
```
User submits registration form
    ↓
Email verification OTP sent (5-digit, 10-min expiry)
    ↓
User enters OTP on verify page
    ↓
OTP validated from email_verifications table
    ↓
Account created in teachers/admins table
    ↓
Credentials email sent (User ID + PIN)
    ↓
User redirects to login page
```

### 2. Face Capture Flow
```
Teacher/Admin clicks "Capture" button
    ↓
Webcam feed activates (OpenCV)
    ↓
Teacher enters student details (name, email, department, class)
    ↓
50+ face samples captured (OpenCV Haar Cascade detection)
    ↓
Face encodings converted to NumPy array
    ↓
Face data appended to faces_data.pkl
    ↓
Student name appended to names.pkl
    ↓
User record inserted into users table
    ↓
Student added to selected class (class_students table)
    ↓
Success message displayed
```

### 3. Face Recognition & Attendance Flow
```
Teacher/Admin clicks "Recognize" button
    ↓
Webcam feed activates
    ↓
User selects class (for class attendance) or gate (for gate attendance)
    ↓
Frame captured from webcam
    ↓
Faces detected using Haar Cascade
    ↓
Face encoded and compared with faces_data.pkl using KNN
    ↓
KNN predicts student name based on k nearest neighbors
    ↓
Confidence calculated from voting
    ↓
If confidence > threshold:
    ↓
    Attendance saved to attendance_records table
    ↓
    Duplicate check (same student, same day, same type/class)
    ↓
    If not duplicate:
        ↓
        Attendance recorded (date, time, status=Present)
        ↓
        Daily CSV updated (Attendance/Attendance_DD-MM-YYYY.csv)
        ↓
        Success notification displayed
    ↓
    Else:
        ↓
        "Already marked" message displayed
```

### 4. Password/PIN Reset Flow
```
User clicks "Forgot Password/PIN"
    ↓
Enters email address
    ↓
5-digit OTP generated and stored in session
    ↓
OTP email sent (10-minute expiry)
    ↓
User enters OTP on verify page
    ↓
OTP validated from session
    ↓
If valid:
    ↓
    Session flag set (otp_verified=True)
    ↓
    User redirects to reset password/PIN page
    ↓
    User enters new password/PIN
    ↓
    Password hashed (SHA256) or PIN validated (4-6 digits)
    ↓
    Database updated (teachers/admins table)
    ↓
    Success message + redirect to login
```

---

## 🔐 Security Considerations

### ✅ Implemented Security Features
1. **Password Hashing:** SHA256 hashing (should upgrade to bcrypt/argon2)
2. **SQL Injection Prevention:** Parameterized queries
3. **Session Security:** Secure session cookies, 30-day expiry
4. **Email Verification:** OTP-based verification before account activation
5. **CSRF Protection:** Flask default CSRF protection
6. **Role-Based Access Control:** Decorator-based route protection
7. **OTP Expiry:** 10-minute expiry for OTP codes
8. **Duplicate Prevention:** Database-level duplicate checks for attendance

### ⚠️ Security Risks & Recommendations
1. **Hardcoded Credentials:** Move SMTP and Cloudinary credentials to environment variables
2. **Debug Mode:** Disable `app.debug = True` in production
3. **Weak Password Hashing:** Upgrade from SHA256 to bcrypt or argon2
4. **No HTTPS:** Deploy with HTTPS/TLS in production
5. **No Rate Limiting:** Add rate limiting for login/OTP endpoints
6. **No Input Validation:** Add server-side validation for all inputs
7. **No File Upload Validation:** Validate file types/sizes for profile images
8. **Session Secret:** Use strong random secret key (currently in `.flask_secret`)

### Recommended Security Enhancements
```python
# 1. Use environment variables
import os
from dotenv import load_dotenv
load_dotenv()

# 2. Add bcrypt password hashing
from flask_bcrypt import Bcrypt
bcrypt = Bcrypt(app)
password_hash = bcrypt.generate_password_hash(password).decode('utf-8')

# 3. Add rate limiting
from flask_limiter import Limiter
limiter = Limiter(app, key_func=lambda: request.remote_addr)
@limiter.limit("5 per minute")
def login():
    pass

# 4. Add input validation
from wtforms import Form, StringField, validators
class RegistrationForm(Form):
    email = StringField('Email', [validators.Email()])
    
# 5. Add HTTPS redirect
from flask_talisman import Talisman
Talisman(app, force_https=True)
```

---

## 📈 Performance Optimization

### Current Performance
- **Face Recognition Speed:** ~100-200ms per frame (depends on hardware)
- **Database Queries:** Direct SQLite queries (no ORM overhead)
- **Face Data Size:** ~500KB for 100 users (50 samples each)
- **Page Load:** Fast (minimal JavaScript, no heavy frameworks)

### Optimization Recommendations
1. **Database Indexing:** Add indexes on frequently queried columns (date, student_name, class_id)
2. **Face Data Caching:** Cache KNN model in memory (reload only on new registration)
3. **Lazy Loading:** Implement pagination for large attendance tables
4. **Image Optimization:** Compress profile images before Cloudinary upload
5. **CDN:** Use Cloudinary CDN for static asset delivery
6. **Database Migration:** Consider PostgreSQL for large-scale deployments
7. **Async Processing:** Use Celery for long-running tasks (email sending, face training)

---

## 🐛 Known Issues & Limitations

### Current Limitations
1. **Monolithic Architecture:** All code in single `app.py` file (~6,000 lines)
2. **No Background Jobs:** Email sending blocks request thread
3. **No API Documentation:** No Swagger/OpenAPI documentation
4. **No Unit Tests:** No automated testing
5. **No Logging:** Limited error logging/monitoring
6. **No Backup System:** No automated database backups
7. **Windows-Only:** `pywin32` dependency limits cross-platform support
8. **Face Recognition Accuracy:** Depends on lighting, camera quality, face angle
9. **No Mobile App:** Web-only interface
10. **Single Instance:** No multi-tenancy support

### Potential Improvements
- Refactor into modular structure (blueprints, services, models)
- Add Celery for background tasks
- Add API documentation (Swagger UI)
- Add pytest unit tests and integration tests
- Add structured logging (logging module with file/console handlers)
- Add automated daily database backups
- Make cross-platform compatible (remove pywin32 or make optional)
- Add face anti-spoofing (liveness detection)
- Develop mobile app (React Native/Flutter)
- Add multi-tenancy (institution isolation)

---

## 📞 Support & Maintenance

### Developer Contact
**Name:** Manish Sharma  
**Email:** mpandat0052@gmail.com  
**Project:** Smart Attendance System v2.0

### Maintenance Guidelines
1. **Database Backups:** Backup `attendance.db` and `db.sqlite3` daily
2. **Face Data Backups:** Backup `data/faces_data.pkl` and `data/names.pkl` weekly
3. **CSV Exports:** Archive old CSV files from `Attendance/` folder monthly
4. **Logs:** Monitor Flask logs for errors
5. **Cloudinary:** Monitor image storage usage (free tier has limits)
6. **SMTP:** Monitor email sending limits (Gmail has daily limits)

---

## 📝 License

This project is licensed under a custom license. See [LICENSE](LICENSE) file for details.

**Copyright © 2025-2026 Manish Sharma. All Rights Reserved.**

Unauthorized copying, modification, distribution, or commercial use is prohibited without explicit written permission.

---

## 🙏 Acknowledgments

- **OpenCV:** Face detection library
- **Scikit-learn:** KNN classifier implementation
- **Flask:** Web framework
- **Bootstrap:** CSS framework
- **Cloudinary:** Cloud image storage
- **Gmail SMTP:** Email delivery service
- **Haar Cascade Model:** Pre-trained face detection model

---

## 📚 Additional Resources

### Documentation Files
- `README.md` — Project overview and quick start guide
- `PROJECT_STRUCTURE.md` — This comprehensive documentation file
- `LICENSE` — Custom project license
- `schema.sql` — Database schema reference
- `requirements.txt` — Python dependencies

### Code Comments
- Main application code is documented inline in `app.py`
- Route functions have docstring comments
- Complex logic sections have explanatory comments

### External References
- [Flask Documentation](https://flask.palletsprojects.com/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [Scikit-learn KNN](https://scikit-learn.org/stable/modules/neighbors.html)
- [Cloudinary Python SDK](https://cloudinary.com/documentation/python_integration)

---

**Last Updated:** June 2026  
**Document Version:** 1.0  
**Project Version:** 2.0
