# 🎓 Smart Attendance System — Face Recognition-Based Attendance Tracker

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-red.svg)](https://opencv.org/)
[![SQLite](https://img.shields.io/badge/Database-SQLite-orange.svg)](https://www.sqlite.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Repository:** [Smart Attendence System](https://github.com/themanishpndt/Face-Recognition-Attendance-System)  
**Primary Language:** Python (Flask)  
**Last Updated:** 2026-02-14  

---

## 📖 Project Overview

The **Smart Attendance System** is a production-ready, self-hosted face recognition attendance tracker built with Flask and OpenCV. It automates attendance marking by recognizing faces in real time, eliminating manual roll calls and preventing proxy attendance.

### 🔑 Key Benefits
- **Contactless & Hygienic** – No physical touchpoints.  
- **Prevents Proxy Attendance** – Each face is uniquely identified.  
- **Real-Time Logging** – Attendance stored in SQLite and daily CSV files.  
- **Secure & Scalable** – Role-based access, password hashing, and cloud-ready architecture.  
- **Cost-Effective** – Open-source alternative to proprietary biometric systems.

### 🎯 Target Users
- Educational institutions (schools, colleges, coaching centers)  
- Corporate offices (employee time tracking)  
- Event organizers (entry management)  
- Security-sensitive areas (access control)  
- Developers and researchers extending face recognition systems

### 🌍 Market Relevance
With the rise of contactless solutions post-pandemic, automated face recognition attendance systems are in high demand. This project provides a robust, extensible baseline that can be adapted to various environments.

---

## ✨ Core Features

- **Real-Time Face Detection & Recognition** – Uses OpenCV Haar cascade for detection and custom encoding/matching for recognition.  
- **Role-Based Access** – Separate dashboards for Admin, Teacher, and Student with appropriate permissions.  
- **Automated Attendance Logging** – Records stored in SQLite (`attendance_records`) and as daily CSV files under `Attendance/`.  
- **User Management** – Register, view, edit, and delete users via web interface.  
- **Face Registration (Capture)** – Capture multiple face samples via webcam; encodings stored in `data/faces_data.pkl`.  
- **Settings Persistence** – Camera index, recognition threshold, sample count saved in `data/settings.pkl`.  
- **Export Reports** – Download attendance as CSV/Excel with date filters.  
- **Responsive Web UI** – Bootstrap 4, custom CSS/JS, mobile-friendly.  
- **Secure Authentication** – Passwords hashed (bcrypt recommended), session management.  
- **Extensible Architecture** – Modular design allows swapping recognition models or databases.

---

## ⚡ System Workflow

```mermaid
flowchart TD
    A[User Face] --> B[Camera Capture]
    B --> C[OpenCV Haarcascade Detection]
    C --> D[Feature Extraction & Encoding]
    D --> E[Compare with Stored Database]
    E -->|Match| F[Attendance Marked ✅]
    E -->|No Match| G[Reject/Registration ❌]
    F --> H[Attendance Stored in DB & CSV]
    H --> I[Reports for Admin/Teacher]
```

---

## 🛠️ Tech Stack

| Layer         | Technologies                                                                 |
|---------------|------------------------------------------------------------------------------|
| **Frontend**  | HTML5, CSS3, JavaScript, Bootstrap 4, Font Awesome                           |
| **Backend**   | Python 3.8+, Flask, Jinja2, Gunicorn                                         |
| **Computer Vision** | OpenCV (Haar cascade), NumPy, Pickle (serialization)                   |
| **Database**  | SQLite (default), extensible to MySQL/PostgreSQL                             |
| **Security**  | bcrypt (or SHA256), Flask sessions, environment variables                    |
| **DevOps**    | Git, Docker, Nginx, Heroku/Railway/AWS                                       |

---

## 🏗 System Architecture

### High-Level Design

```
+----------------------+      +--------------------+      +--------------------+
|  Browser (Client)    | <--> |  Flask App (app.py) | <--> |   SQLite DB        |
|  - Jinja2 Templates  |      |  - Routes & Auth    |      |   attendance.db    |
|  - JS (capture.js)   |      |  - Face Recognition |      +--------------------+
|  - CSS               |      |  - CSV Export       |
+----------------------+      +--------------------+
              |                             |
              |                             v
              |                      +----------------+
              |                      |  /data/ (Pickle)|
              |                      | faces_data.pkl, |
              |                      | names.pkl, etc. |
              |                      +----------------+
```

### Backend Structure
- **app.py** – Single-entry Flask application handling routing, authentication, face processing, database operations, and CSV exports.
- **templates/** – Jinja2 HTML templates for all pages (auth, capture, recognize, attendance, teacher views).
- **static/** – CSS, JavaScript, and image assets.
- **data/** – Pickled face encodings, names, recognizer, and system settings.
- **Attendance/** – Daily CSV attendance logs.

### Frontend Components
- Server-rendered pages using Jinja2.
- Client-side JavaScript (`capture.js`, `recognize.js`) for camera access, frame capture, and streaming.
- Responsive design with Bootstrap and custom CSS.

---

## 🔄 Workflow Explanation

### Step-by-Step User Journey

1. **Registration (Teacher/Admin)**
   - Navigate to `/register`, provide username, email, password.
   - After login, teacher/admin can create classes and manage students.

2. **Student Face Registration**
   - Open `/capture` page.
   - Capture multiple face samples (e.g., 50 images) via webcam.
   - Server detects faces, computes encodings, and stores them in `data/faces_data.pkl` and user metadata in the `users` table.

3. **Attendance Marking (Recognition)**
   - Open `/recognize` page.
   - Live webcam stream sends frames to server.
   - Server detects faces, compares against stored encodings.
   - On match, attendance is recorded in `attendance_records` and appended to the daily CSV in `Attendance/`.

4. **Reporting**
   - Admin/Teacher views attendance logs via `/attendance`.
   - Export filtered data via `/export_attendance` as CSV/Excel.

### Data Processing Pipeline
- Frame capture → Face detection (Haar cascade) → Face ROI extraction → Encoding (e.g., raw pixels or embedding) → Matching (nearest neighbor) → Attendance record creation (DB + CSV).

### Authentication Flow
- Session-based authentication using Flask.
- Passwords hashed with bcrypt (recommended).
- Role checks on routes to restrict access.

---

## 🧠 Algorithms Used

### Face Detection
- **OpenCV Haar Cascade Classifier** – Pre-trained model `haarcascade_frontalface_default.xml` detects frontal faces.
- **Preprocessing**: Convert frame to grayscale, apply histogram equalization, detect faces, and resize ROI to fixed dimensions (e.g., 100x100).

### Face Recognition
- **Encoding Method**: Simple pixel vector or a lightweight embedding (e.g., using OpenCV LBPH or a custom CNN). The project stores encodings in `faces_data.pkl` and corresponding names in `names.pkl`.
- **Matching**: Euclidean distance or cosine similarity between the encoding of the detected face and all stored encodings.
- **Threshold**: If the smallest distance < `RECOGNITION_THRESHOLD`, the face is recognized; otherwise, rejected.

### Attendance Marking Logic
```python
def mark_attendance(name, class_id, teacher_id):
    date = datetime.now().strftime("%d-%m-%Y")
    if not already_marked_today(name, class_id, date):
        time = datetime.now().strftime("%H:%M:%S")
        insert_into_db(name, class_id, teacher_id, date, time, "Present")
        append_to_csv(f"Attendance/Attendance_{date}.csv", [name, date, time, "Present"])
```

### Business Rules
- Prevent duplicate attendance for the same student in the same day/class.
- If a student is not recognized, they can be redirected to registration.

---

## 📁 Project Structure (Detailed)

```
Smart Attendence System/
│
├── app.py                         # Main Flask application
├── schema.sql                     # SQL schema to create tables
├── attendance.db                  # SQLite database (auto-created)
├── haarcascade_frontalface_default.xml  # Face detection model
├── requirements.txt               # Python dependencies
├── README.md                      # This file
│
├── Attendance/                    # Daily CSV attendance logs
│   ├── Attendance_07-02-2026.csv
│   ├── Attendance_08-02-2026.csv
│   └── ...
│
├── data/                          # Persistent data files
│   ├── faces_data.pkl             # List of face encodings
│   ├── names.pkl                  # Corresponding names
│   ├── face_recognizer.pkl        # Optional trained recognizer (e.g., LBPH)
│   ├── settings.pkl               # Camera index, threshold, sample count
│   └── today_attendance_26-04-2025.csv  # Temporary current-day attendance
│
├── static/                        # Static assets
│   ├── bg.png
│   ├── css/
│   │   ├── animations.css
│   │   ├── capture.css
│   │   ├── attendance.css
│   │   └── main.css
│   └── js/
│       ├── capture.js
│       ├── dashboard.js
│       ├── attendance.js
│       └── main.js
│
└── templates/                     # Jinja2 HTML templates
    ├── base.html                  # Base layout with navbar
    ├── index.html                 # Landing page
    ├── capture.html               # Face capture page
    ├── recognize.html             # Face recognition page
    ├── attendance.html            # Attendance dashboard
    ├── export_attendance.html     # Report export page
    ├── manage_users.html          # User management (admin)
    ├── settings.html              # System settings
    ├── error.html                  # Error display
    ├── instructions.html           # Help/instructions
    ├── result.html                 # Result page after capture/recognition
    ├── auth/
    │   ├── login.html
    │   └── register.html
    └── teacher/
        ├── dashboard.html
        ├── classes.html
        ├── class_detail.html
        └── student_attendance.html
```

---

## 🗄️ Database Schema (SQLite)

### Table: `teachers`
| Column        | Type      | Description                         |
|---------------|-----------|-------------------------------------|
| id            | INTEGER   | Primary key, autoincrement          |
| username      | TEXT      | Unique login name                   |
| password_hash | TEXT      | Hashed password (bcrypt recommended)|
| email         | TEXT      | Unique email address                |
| full_name     | TEXT      | Display name                        |
| created_at    | TIMESTAMP | Registration timestamp              |

### Table: `classes`
| Column      | Type      | Description                         |
|-------------|-----------|-------------------------------------|
| id          | INTEGER   | Primary key, autoincrement          |
| teacher_id  | INTEGER   | Foreign key → teachers.id           |
| name        | TEXT      | Class name (e.g., "Math 101")       |
| description | TEXT      | Optional description                |
| created_at  | TIMESTAMP | Creation time                       |

### Table: `class_students`
| Column       | Type      | Description                         |
|--------------|-----------|-------------------------------------|
| class_id     | INTEGER   | Foreign key → classes.id            |
| student_name | TEXT      | Name of student (matches `users.name`)|
| added_at     | TIMESTAMP | When student was added to class     |

### Table: `attendance_records`
| Column       | Type      | Description                         |
|--------------|-----------|-------------------------------------|
| id           | INTEGER   | Primary key, autoincrement          |
| student_name | TEXT      | Name of student                     |
| class_id     | INTEGER   | Foreign key → classes.id            |
| teacher_id   | INTEGER   | Foreign key → teachers.id (optional)|
| date         | TEXT      | Date in DD-MM-YYYY format           |
| time         | TEXT      | Time in HH:MM:SS format             |
| status       | TEXT      | 'Present' or 'Absent'               |
| notes        | TEXT      | Optional notes                      |

### Table: `users` (Face Recognition Users)
| Column     | Type      | Description                         |
|------------|-----------|-------------------------------------|
| username   | TEXT      | Primary key                         |
| name       | TEXT      | Full name                           |
| email      | TEXT      | Email address                       |
| user_id    | TEXT      | Unique identifier (e.g., roll number)|
| department | TEXT      | Department/branch                   |
| phone      | TEXT      | Contact number                      |
| role       | TEXT      | 'student' or 'teacher'               |
| notes      | TEXT      | Extra information                   |
| created_at | TIMESTAMP | Registration time                   |

---

## 🖥️ UI/UX & Backend Integration

- **Responsive Design**: All pages use Bootstrap 4 with custom CSS for a modern look.
- **Dashboards**: Card-based layouts with real-time stats (total users, today's attendance, etc.).
- **Flash Messages**: Success/error/info messages for all user actions.
- **Live Search & Filtering**: On manage users and attendance pages.
- **Export Options**: CSV, Excel, PDF (via additional libraries if needed).
- **Secure Authentication**: Login/register with hashed passwords; session timeout.
- **All Forms Validated**: Client-side (JavaScript) and server-side (Flask).
- **Static Assets**: Organized under `static/` for easy CDN integration.

---

## 🔄 Data Flow & Major Features

1. **Face Registration**
   - User accesses `/capture`, grants camera permission.
   - JavaScript captures frames, sends to `/capture` endpoint.
   - Server processes each frame: detects face, extracts encoding, and stores in `faces_data.pkl` and `names.pkl`.
   - User details saved in `users` table.

2. **Attendance Marking**
   - User accesses `/recognize`, camera streams frames.
   - Server receives frames, detects face, compares encodings.
   - On match, attendance logged in `attendance_records` and daily CSV.
   - Response returns student name and status.

3. **User Management**
   - Admin/teacher views `/manage_users`; can search, edit, or delete users.
   - All user info retrieved from `users` table.

4. **Statistics Dashboard**
   - `/attendance` shows total records, unique students, today's count.
   - Charts may be added (e.g., using Chart.js).

5. **Settings**
   - `/settings` allows configuration of camera index, recognition threshold, sample count, etc.
   - Settings persisted in `data/settings.pkl`.

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.8+
- pip
- Webcam
- Git
- Virtual environment (recommended)

### Step 1: Clone Repository
```bash
git clone https://github.com/themanishpndt/Face-Recognition-Attendance-System.git
cd Face-Recognition-Attendance-System
```

### Step 2: Create Virtual Environment
**Windows (PowerShell)**
```powershell
python -m venv venv
venv\Scripts\Activate
```
**macOS / Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

Typical `requirements.txt`:
```
Flask>=2.0
opencv-python>=4.5
numpy
bcrypt
python-dotenv
gunicorn
```

### Step 4: Initialize Database
```bash
sqlite3 attendance.db < schema.sql
```
(Or run the app once; it may auto-create tables if configured.)

### Step 5: Configure Environment Variables
Create a `.env` file in the project root:
```
FLASK_APP=app.py
FLASK_ENV=development
SECRET_KEY=your-strong-secret-key
DATABASE_URL=sqlite:///attendance.db
RECOGNITION_THRESHOLD=0.6
CAMERA_INDEX=0
```

### Step 6: Run Application
```bash
python app.py
# or
flask run
```

### Step 7: Access in Browser
```
http://127.0.0.1:5000/
```

---

## 🗂 Configuration & Data Files

- **`data/faces_data.pkl`** – Pickled list/array of face encodings.
- **`data/names.pkl`** – Pickled list of names corresponding to encodings.
- **`data/face_recognizer.pkl`** – Optional trained recognizer (e.g., LBPH model) for faster matching.
- **`data/settings.pkl`** – Dictionary with keys: `camera_index`, `recognition_threshold`, `sample_count`, `notifications`.
- **`Attendance/Attendance_*.csv`** – Daily attendance files with format: `Student_ID,Name,Date,Time,Status`.

If these files are missing, the app will start with empty state (no registered faces) and default settings.

---

## 🌐 Routes & Pages (Typical)

| Route                 | Method | Description                          | Auth Required |
|-----------------------|--------|--------------------------------------|---------------|
| `/`                   | GET    | Landing page (index.html)            | No            |
| `/login`              | GET/POST | User login                          | No            |
| `/register`           | GET/POST | New user registration                | No            |
| `/logout`             | GET    | Log out                              | Yes           |
| `/capture`            | GET    | Face capture page                    | Yes           |
| `/capture`            | POST   | Receive captured frames              | Yes           |
| `/recognize`          | GET    | Face recognition page                | Yes           |
| `/recognize`          | POST   | Receive frame and return recognition | Yes           |
| `/attendance`         | GET    | Attendance dashboard                 | Yes (teacher/admin) |
| `/export_attendance`  | GET    | Export attendance (CSV/Excel)        | Yes (teacher/admin) |
| `/manage_users`       | GET/POST | User management (admin only)       | Yes (admin)   |
| `/settings`           | GET/POST | System settings                    | Yes (admin)   |
| `/teacher/dashboard`  | GET    | Teacher dashboard                    | Yes (teacher) |
| `/teacher/classes`    | GET    | List classes                         | Yes (teacher) |
| `/teacher/class/<id>` | GET    | Class details and attendance         | Yes (teacher) |
| `/static/<path>`      | GET    | Serve static files                   | No            |

---

## 🧪 API Documentation (Inferred)

The project is primarily server-rendered, but the following endpoints act as a de facto API for AJAX calls.

### `POST /capture`
- **Request**: Multipart/form-data or base64-encoded image.
- **Response**: JSON `{ "success": true, "user_id": "..." }`

### `POST /recognize`
- **Request**: JSON `{ "image": "base64-string" }`
- **Response**: JSON `{ "success": true, "match": true, "name": "John Doe", "confidence": 0.12, "recorded": true }` or `{ "match": false }`

### `GET /attendance/export`
- **Query Params**: `?date=DD-MM-YYYY&class_id=1`
- **Response**: CSV file download.

### `POST /manage_users/delete`
- **Request**: JSON `{ "username": "john" }`
- **Response**: JSON `{ "success": true }`

All non-GET endpoints require authentication (session cookie).

---

## 🔒 Security Implementation

- **Authentication**: Session-based with Flask. Passwords hashed using `bcrypt` (recommended) or SHA256.
- **Authorization**: Role checks (`current_user.role`) on routes; templates conditionally show admin/teacher actions.
- **Data Protection**:
  - Face encodings stored as pickled files – consider encryption at rest in production.
  - Database passwords and secret keys stored in environment variables.
- **CSRF Protection**: Implement using Flask-WTF or custom token validation for forms.
- **HTTPS**: Enforce in production via reverse proxy (Nginx).
- **Input Validation**: Server-side validation for all form inputs.
- **Logging**: Admin actions logged for audit (optional).

### Recommendations for Production
- Replace SQLite with PostgreSQL.
- Store face embeddings in encrypted form.
- Use a secrets manager for environment variables.
- Enable rate limiting on recognition endpoints.

---

## 🚀 Deployment Guide

### Local Development
```bash
python app.py
```

### Production with Gunicorn & Nginx
1. Install Gunicorn: `pip install gunicorn`
2. Run Gunicorn:
   ```bash
   gunicorn --bind 127.0.0.1:5000 app:app
   ```
3. Configure Nginx as reverse proxy:
   ```nginx
   server {
       listen 80;
       server_name example.com;
       location / {
           proxy_pass http://127.0.0.1:5000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
       location /static/ {
           alias /path/to/Smart Attendence System/static/;
       }
   }
   ```
4. Set up SSL with Let's Encrypt.

### Docker Deployment
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV FLASK_ENV=production
ENV DATABASE_URL=sqlite:///attendance.db
EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
```

Build and run:
```bash
docker build -t attendance-system .
docker run -p 5000:5000 attendance-system
```

### Cloud Deployment
- **Heroku**: Add `Procfile` with `web: gunicorn app:app`, set environment variables.
- **AWS**: Use Elastic Beanstalk or EC2 with Nginx.
- **Railway/Render**: Point to repository, auto-deploy.

---

## 📈 Scalability Plan

- **Horizontal Scaling**: Run multiple Gunicorn workers behind a load balancer.
- **Database**: Migrate from SQLite to PostgreSQL for concurrent writes.
- **Recognition Microservice**: Offload face encoding/matching to a dedicated service (GPU-enabled) using Celery or gRPC.
- **Vector Database**: Store embeddings in FAISS or Milvus for fast similarity search.
- **Caching**: Cache frequent recognitions with Redis.
- **CDN**: Serve static assets via CDN.

---

## 🔮 Future Enhancements

- **Deep Learning Models**: Replace Haar cascade with MTCNN or YOLO for face detection; use FaceNet/ArcFace for robust embeddings.
- **Anti-Spoofing**: Liveness detection to prevent photo/video attacks.
- **Mobile App**: Native Android/iOS app with on-device recognition.
- **Integration**: Connect with school MIS, HR systems via REST API.
- **Multi-Face Handling**: Recognize multiple faces in a single frame.
- **Emotion/Engagement Analysis**: Additional analytics for classrooms.
- **GDPR Compliance**: Data retention policies, consent management, encryption at rest.

---

## 📜 Limitations

- **Lighting Conditions**: Accuracy decreases in poor lighting.
- **Camera Quality**: Low-resolution cameras may affect detection.
- **Hardware Dependency**: Performance relies on CPU/GPU.
- **Privacy Concerns**: Biometric data requires careful handling.
- **Single-Face Focus**: Currently optimized for one face per frame.

---

## 📌 Applications

- **Educational Institutions**: Student attendance in classrooms.
- **Corporate Offices**: Employee check-in/out.
- **Events**: Entry management and attendee tracking.
- **Healthcare**: Staff attendance logging in hospitals.
- **Security**: Access control in restricted areas.

---

## 🧪 Testing

- **Unit Testing**: Test individual functions (face encoding, DB operations) using `pytest`.
- **Integration Testing**: Test recognition pipeline with sample images.
- **System Testing**: End-to-end workflow from registration to attendance export.
- **Manual Testing**: Verify UI responsiveness and camera functionality.

---

## 📄 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request.

### Guidelines
- Keep changes focused and small.
- Write clear commit messages.
- Add tests for new functionality.
- Update documentation accordingly.

---

## 👨‍💻 Author

**Manish Sharma**  
📍 Ghaziabad, Uttar Pradesh, India  
📞 +91 7982682852  
📧 [manishsharma93155@gmail.com](mailto:manishsharma93155@gmail.com)  
🔗 [LinkedIn](https://www.linkedin.com/in/themanishpndt)  
💻 [GitHub](https://github.com/themanishpndt)  
🌐 [Portfolio](https://themanishpndt.github.io/Portfolio/)

---

## 📎 Appendix

### Recommended Environment Variables
```
FLASK_APP=app.py
FLASK_ENV=production
SECRET_KEY=<long-random-string>
DATABASE_URL=sqlite:///attendance.db   # or postgresql://user:pass@host/db
RECOGNITION_THRESHOLD=0.6
CAMERA_INDEX=0
SMTP_HOST=smtp.example.com   # for email notifications (optional)
SMTP_PORT=587
SMTP_USER=user@example.com
SMTP_PASSWORD=strongpassword
```

### Quick Operational Checklist
- [ ] Ensure `Attendance/` and `data/` directories are writable.
- [ ] Verify `haarcascade_frontalface_default.xml` exists and is readable.
- [ ] Confirm `requirements.txt` is up-to-date.
- [ ] Set strong `SECRET_KEY` in production.
- [ ] Enable HTTPS in production.
- [ ] Backup `data/*.pkl` and database regularly.

---

If you find this project useful, please ⭐ the repository on GitHub! 🙌