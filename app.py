import cv2
import pickle
import numpy as np
import os
import csv
import time
import sqlite3
import hashlib
import secrets
import smtplib
import random
import string
import base64
import cloudinary
import cloudinary.uploader
from functools import wraps
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from win32com.client import Dispatch
from sklearn.neighbors import KNeighborsClassifier
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    send_from_directory,
    session,
    flash,
    jsonify,
    make_response,
    Response,
)

# Initialize Flask app
app = Flask(__name__)

# Persistent secret key — survives server restarts so sessions stay valid
_secret_key_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".flask_secret")
if os.path.exists(_secret_key_file):
    with open(_secret_key_file, "r") as _f:
        app.secret_key = _f.read().strip()
else:
    app.secret_key = secrets.token_hex(32)
    with open(_secret_key_file, "w") as _f:
        _f.write(app.secret_key)

app.permanent_session_lifetime = timedelta(days=30)  # "Remember me" sessions last 30 days


# ═══════════════════════════════════════════
# Cloudinary Configuration (persistent media storage)
# ═══════════════════════════════════════════
cloudinary.config(
    cloud_name="dud3f00ay",
    api_key="764652939378289",
    api_secret="gu7Rwmz8jB4I0vsI3VYZNC3Ri0Q",
    secure=True,
)


def upload_profile_image(file, folder="profile_images"):
    """Upload an image to Cloudinary and return the secure URL."""
    try:
        result = cloudinary.uploader.upload(
            file,
            folder=folder,
            transformation=[
                {"width": 400, "height": 400, "crop": "fill", "gravity": "face"},
                {"quality": "auto", "fetch_format": "auto"},
            ],
            overwrite=True,
            resource_type="image",
        )
        return result.get("secure_url")
    except Exception as e:
        print(f"Cloudinary upload error: {e}")
        return None


# ═══════════════════════════════════════════
# Auth Decorators & Context Processor
# ═══════════════════════════════════════════
def teacher_login_required(f):
    """Decorator that redirects to login if teacher is not authenticated."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "teacher_id" not in session:
            flash("Please log in first.", "warning")
            return redirect(url_for("teacher_login"))
        return f(*args, **kwargs)
    return decorated_function


def admin_login_required(f):
    """Decorator that redirects to admin login if admin is not authenticated."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "admin_id" not in session:
            flash("Admin login required.", "warning")
            return redirect(url_for("admin_login"))
        return f(*args, **kwargs)
    return decorated_function


def any_login_required(f):
    """Decorator that requires either admin or teacher to be logged in."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "admin_id" not in session and "teacher_id" not in session:
            flash("Please sign in first.", "warning")
            return redirect(url_for("home"))
        return f(*args, **kwargs)
    return decorated_function


@app.context_processor
def inject_auth_info():
    """Make auth info available in ALL templates automatically."""
    return {
        "teacher_logged_in": "teacher_id" in session,
        "current_teacher_name": session.get("teacher_name", ""),
        "current_teacher_username": session.get("teacher_username", ""),
        "current_teacher_image": session.get("teacher_profile_image", ""),
        "admin_logged_in": "admin_id" in session,
        "current_admin_name": session.get("admin_name", ""),
        "current_admin_username": session.get("admin_username", ""),
        "current_admin_image": session.get("admin_profile_image", ""),
    }


# ═══════════════════════════════════════════
# SMTP Email Configuration
# ═══════════════════════════════════════════
SMTP_EMAIL = "mpandat0052@gmail.com"
SMTP_PASSWORD = "delq fqbt nxgm kped"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
APP_NAME = "Smart Attendance System"


def send_email(to_email, subject, html_body):
    """Send an email using Gmail SMTP."""
    try:
        msg = MIMEMultipart("alternative")
        msg["From"] = f"{APP_NAME} <{SMTP_EMAIL}>"
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(html_body, "html"))

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_EMAIL, SMTP_PASSWORD)
            server.sendmail(SMTP_EMAIL, to_email, msg.as_string())
        print(f"Email sent to {to_email}: {subject}")
        return True
    except Exception as e:
        print(f"Email sending failed: {str(e)}")
        return False


def generate_otp(length=6):
    """Generate a random numeric OTP."""
    return "".join(random.choices(string.digits, k=length))


def generate_token():
    """Generate a random URL-safe token."""
    return secrets.token_urlsafe(32)


def build_verification_email(full_name, otp_code):
    """Build the HTML body for account verification email (Teacher - purple)."""
    return f"""
    <div style="font-family:'Segoe UI',Arial,sans-serif;max-width:560px;margin:0 auto;background:#fff;border-radius:16px;overflow:hidden;box-shadow:0 8px 32px rgba(102,126,234,.18);">
      <div style="background:linear-gradient(135deg,#667eea,#764ba2);padding:2.2rem 2rem;text-align:center;color:#fff;">
        <div style="width:64px;height:64px;background:rgba(255,255,255,.18);border-radius:50%;margin:0 auto 1rem;display:flex;align-items:center;justify-content:center;font-size:1.7rem;">&#128274;</div>
        <h2 style="margin:0;font-size:1.5rem;font-weight:800;">Email Verification</h2>
        <p style="opacity:.85;margin:.4rem 0 0;font-size:.88rem;">Smart Attendance System — Teacher Portal</p>
      </div>
      <div style="padding:2rem;text-align:center;">
        <p style="color:#444;font-size:.95rem;line-height:1.6;">Hello <strong>{full_name}</strong>,</p>
        <p style="color:#555;font-size:.92rem;line-height:1.6;">Thank you for registering. Please use the verification code below to complete your account setup:</p>
        <div style="background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;font-size:2.2rem;font-weight:800;letter-spacing:10px;padding:1.1rem 2.5rem;border-radius:14px;display:inline-block;margin:1.2rem 0;box-shadow:0 4px 16px rgba(102,126,234,.3);">{otp_code}</div>
        <div style="background:#f8f9ff;border:1px solid #e8ecff;border-radius:10px;padding:.8rem 1rem;margin:1rem auto;max-width:380px;">
          <p style="color:#667eea;font-size:.82rem;margin:0;font-weight:600;">&#9200; This code expires in <strong>10 minutes</strong></p>
        </div>
        <p style="color:#999;font-size:.8rem;margin-top:.8rem;">If you didn't create an account, you can safely ignore this email.</p>
      </div>
      <div style="background:#f4f3ff;padding:1rem;text-align:center;font-size:.75rem;color:#aaa;border-top:1px solid #e8e6ff;">
        &copy; 2025 Smart Attendance System &middot; All rights reserved
      </div>
    </div>
    """


def build_admin_verification_email(full_name, otp_code):
    """Build the HTML body for account verification email (Admin - pink)."""
    return f"""
    <div style="font-family:'Segoe UI',Arial,sans-serif;max-width:560px;margin:0 auto;background:#fff;border-radius:16px;overflow:hidden;box-shadow:0 8px 32px rgba(240,147,251,.18);">
      <div style="background:linear-gradient(135deg,#f093fb,#f5576c);padding:2.2rem 2rem;text-align:center;color:#fff;">
        <div style="width:64px;height:64px;background:rgba(255,255,255,.18);border-radius:50%;margin:0 auto 1rem;display:flex;align-items:center;justify-content:center;font-size:1.7rem;">&#128737;</div>
        <h2 style="margin:0;font-size:1.5rem;font-weight:800;">Admin Email Verification</h2>
        <p style="opacity:.85;margin:.4rem 0 0;font-size:.88rem;">Smart Attendance System — Admin Portal</p>
      </div>
      <div style="padding:2rem;text-align:center;">
        <p style="color:#444;font-size:.95rem;line-height:1.6;">Hello <strong>{full_name}</strong>,</p>
        <p style="color:#555;font-size:.92rem;line-height:1.6;">Thank you for registering as an administrator. Please enter the verification code below to activate your admin account:</p>
        <div style="background:linear-gradient(135deg,#f093fb,#f5576c);color:#fff;font-size:2.2rem;font-weight:800;letter-spacing:10px;padding:1.1rem 2.5rem;border-radius:14px;display:inline-block;margin:1.2rem 0;box-shadow:0 4px 16px rgba(245,87,108,.3);">{otp_code}</div>
        <div style="background:#fff5f7;border:1px solid #fce;border-radius:10px;padding:.8rem 1rem;margin:1rem auto;max-width:380px;">
          <p style="color:#f5576c;font-size:.82rem;margin:0;font-weight:600;">&#9200; This code expires in <strong>10 minutes</strong></p>
        </div>
        <p style="color:#999;font-size:.8rem;margin-top:.8rem;">If you didn't request this, you can safely ignore this email.</p>
      </div>
      <div style="background:#fff5f7;padding:1rem;text-align:center;font-size:.75rem;color:#aaa;border-top:1px solid #fce;">
        &copy; 2025 Smart Attendance System &middot; All rights reserved
      </div>
    </div>
    """


def build_reset_email(full_name, otp_code):
    """Build the HTML body for password reset OTP email (pink theme)."""
    return f"""
    <div style="font-family:'Segoe UI',Arial,sans-serif;max-width:520px;margin:0 auto;background:#fff;border-radius:16px;overflow:hidden;box-shadow:0 4px 24px rgba(240,147,251,.15);">
      <div style="background:linear-gradient(135deg,#f093fb,#f5576c);padding:2rem;text-align:center;color:#fff;">
        <div style="width:56px;height:56px;background:rgba(255,255,255,.2);border-radius:50%;margin:0 auto 1rem;display:flex;align-items:center;justify-content:center;font-size:1.5rem;">&#128272;</div>
        <h2 style="margin:0;font-size:1.4rem;">Reset Your Password</h2>
        <p style="opacity:.85;margin:.3rem 0 0;">Smart Attendance System</p>
      </div>
      <div style="padding:2rem;text-align:center;">
        <p style="color:#555;font-size:.95rem;">Hello <strong>{full_name}</strong>,</p>
        <p style="color:#555;font-size:.95rem;">We received a request to reset your password. Use the verification code below:</p>
        <div style="background:linear-gradient(135deg,#f093fb,#f5576c);color:#fff;font-size:2.2rem;font-weight:800;letter-spacing:10px;padding:1.1rem 2.5rem;border-radius:14px;display:inline-block;margin:1.2rem 0;box-shadow:0 4px 16px rgba(245,87,108,.3);">{otp_code}</div>
        <div style="background:#fff5f7;border:1px solid #fce;border-radius:10px;padding:.8rem 1rem;margin:1rem auto;max-width:380px;">
          <p style="color:#f5576c;font-size:.82rem;margin:0;font-weight:600;">&#9200; This code expires in <strong>10 minutes</strong></p>
        </div>
        <p style="color:#888;font-size:.82rem;">If you didn't request this, your account is safe — just ignore this email.</p>
      </div>
      <div style="background:#fff5f7;padding:1rem;text-align:center;font-size:.75rem;color:#aaa;border-top:1px solid #fce;">Smart Attendance System &copy; 2025</div>
    </div>
    """


def build_pin_reset_email(full_name, otp_code):
    """Build the HTML body for PIN reset OTP email (purple theme)."""
    return f"""
    <div style="font-family:'Segoe UI',Arial,sans-serif;max-width:520px;margin:0 auto;background:#fff;border-radius:16px;overflow:hidden;box-shadow:0 8px 32px rgba(102,126,234,.18);">
      <div style="background:linear-gradient(135deg,#667eea,#764ba2);padding:2rem;text-align:center;color:#fff;">
        <div style="width:56px;height:56px;background:rgba(255,255,255,.2);border-radius:50%;margin:0 auto 1rem;display:flex;align-items:center;justify-content:center;font-size:1.5rem;">&#128273;</div>
        <h2 style="margin:0;font-size:1.4rem;">Reset Your PIN</h2>
        <p style="opacity:.85;margin:.3rem 0 0;">Smart Attendance System</p>
      </div>
      <div style="padding:2rem;text-align:center;">
        <p style="color:#555;font-size:.95rem;">Hello <strong>{full_name}</strong>,</p>
        <p style="color:#555;font-size:.95rem;">We received a request to reset your User PIN. Use the verification code below:</p>
        <div style="background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;font-size:2.2rem;font-weight:800;letter-spacing:10px;padding:1.1rem 2.5rem;border-radius:14px;display:inline-block;margin:1.2rem 0;box-shadow:0 4px 16px rgba(102,126,234,.3);">{otp_code}</div>
        <div style="background:#f8f9ff;border:1px solid #e8ecff;border-radius:10px;padding:.8rem 1rem;margin:1rem auto;max-width:380px;">
          <p style="color:#667eea;font-size:.82rem;margin:0;font-weight:600;">&#9200; This code expires in <strong>10 minutes</strong></p>
        </div>
        <p style="color:#888;font-size:.82rem;">If you didn't request this, your account is safe — just ignore this email.</p>
      </div>
      <div style="background:#f4f3ff;padding:1rem;text-align:center;font-size:.75rem;color:#aaa;border-top:1px solid #e8e6ff;">Smart Attendance System &copy; 2025</div>
    </div>
    """


def build_welcome_email(full_name, email_id, password, role="Admin", user_id="", user_pin=""):
    """Build HTML email sent after successful OTP verification (Admin welcome)."""
    gradient = "linear-gradient(135deg,#f093fb,#f5576c)" if role == "Admin" else "linear-gradient(135deg,#667eea,#764ba2)"
    icon = "&#128737;" if role == "Admin" else "&#128274;"
    pin_section = ""
    if user_id and user_pin:
        pin_section = f"""
        <div style="background:#f0fff4;border:1px solid #c6f6d5;border-radius:12px;padding:1.2rem;margin:1.2rem 0;">
          <p style="color:#276749;font-size:.85rem;font-weight:700;margin:0 0 .6rem;"><i>&#128273;</i> Alternative Login (User ID &amp; PIN)</p>
          <table style="width:100%;border-collapse:collapse;">
            <tr><td style="padding:8px 12px;color:#888;font-size:.85rem;font-weight:600;white-space:nowrap;">User ID</td>
                <td style="padding:8px 12px;font-size:1.05rem;font-weight:800;color:#333;">{user_id}</td></tr>
            <tr><td style="padding:8px 12px;color:#888;font-size:.85rem;font-weight:600;border-top:1px solid #e2e8f0;white-space:nowrap;">User PIN</td>
                <td style="padding:8px 12px;font-size:1.3rem;font-weight:800;color:#333;letter-spacing:6px;border-top:1px solid #e2e8f0;">{user_pin}</td></tr>
          </table>
        </div>"""
    return f"""
    <div style="font-family:'Segoe UI',Arial,sans-serif;max-width:560px;margin:0 auto;background:#fff;border-radius:16px;overflow:hidden;box-shadow:0 4px 24px rgba(0,0,0,.1);">
      <div style="background:{gradient};padding:2rem;text-align:center;color:#fff;">
        <div style="width:60px;height:60px;background:rgba(255,255,255,.2);border-radius:50%;margin:0 auto 1rem;display:flex;align-items:center;justify-content:center;font-size:1.6rem;">{icon}</div>
        <h2 style="margin:0;font-size:1.4rem;">Welcome to Smart Attendance!</h2>
        <p style="opacity:.85;margin:.4rem 0 0;font-size:.9rem;">{role} Account Activated Successfully</p>
      </div>
      <div style="padding:2rem;">
        <p style="color:#555;font-size:.95rem;">Hello <strong>{full_name}</strong>,</p>
        <p style="color:#555;font-size:.95rem;">Your {role.lower()} account has been verified and is now active. Here are your login credentials:</p>
        <div style="background:#f8f9ff;border:1px solid #e8ecff;border-radius:12px;padding:1.2rem;margin:1.2rem 0;">
          <table style="width:100%;border-collapse:collapse;">
            <tr><td style="padding:8px 12px;color:#888;font-size:.85rem;font-weight:600;white-space:nowrap;">Email ID</td>
                <td style="padding:8px 12px;font-size:1.05rem;font-weight:800;color:#333;">{email_id}</td></tr>
            <tr><td style="padding:8px 12px;color:#888;font-size:.85rem;font-weight:600;border-top:1px solid #eef;white-space:nowrap;">Password</td>
                <td style="padding:8px 12px;font-size:1.05rem;font-weight:800;color:#333;border-top:1px solid #eef;">{password}</td></tr>
          </table>
        </div>
        {pin_section}
        <p style="color:#e74c3c;font-size:.82rem;margin:.8rem 0;">&#9888; Please keep your credentials safe and do not share them with anyone.</p>
        <p style="color:#888;font-size:.82rem;">You can now sign in using your email &amp; password, or your User ID &amp; PIN.</p>
      </div>
      <div style="background:#f8f8ff;padding:1rem;text-align:center;font-size:.75rem;color:#aaa;border-top:1px solid #eee;">Smart Attendance System &copy; 2025</div>
    </div>
    """


def build_credentials_email(full_name, user_id, user_pin, role="Teacher", email="", department="", designation="", phone="", admin_name="", admin_email=""):
    """Build HTML email with ALL login details sent when admin creates a teacher/staff account."""
    role_color = "#667eea" if role == "Teacher" else "#f093fb" if role == "Staff" else "#06d6a0"
    gradient = f"linear-gradient(135deg,{role_color},#764ba2)"
    extra_rows = ""
    if email:
        extra_rows += f'<tr><td style="padding:8px 12px;color:#888;font-size:.85rem;font-weight:600;border-top:1px solid #eef;white-space:nowrap;">Email</td><td style="padding:8px 12px;font-size:.95rem;font-weight:700;color:#333;border-top:1px solid #eef;">{email}</td></tr>'
    if department:
        extra_rows += f'<tr><td style="padding:8px 12px;color:#888;font-size:.85rem;font-weight:600;border-top:1px solid #eef;white-space:nowrap;">Department</td><td style="padding:8px 12px;font-size:.95rem;font-weight:700;color:#333;border-top:1px solid #eef;">{department}</td></tr>'
    if designation:
        extra_rows += f'<tr><td style="padding:8px 12px;color:#888;font-size:.85rem;font-weight:600;border-top:1px solid #eef;white-space:nowrap;">Designation</td><td style="padding:8px 12px;font-size:.95rem;font-weight:700;color:#333;border-top:1px solid #eef;">{designation}</td></tr>'
    if phone:
        extra_rows += f'<tr><td style="padding:8px 12px;color:#888;font-size:.85rem;font-weight:600;border-top:1px solid #eef;white-space:nowrap;">Phone</td><td style="padding:8px 12px;font-size:.95rem;font-weight:700;color:#333;border-top:1px solid #eef;">{phone}</td></tr>'
    admin_section = ""
    if admin_name or admin_email:
        admin_section = f"""
        <div style="background:#f0f4ff;border:1px solid #d0d8f0;border-radius:10px;padding:.8rem 1rem;margin:1rem 0;">
          <p style="color:#4a5568;font-size:.82rem;margin:0;"><strong>&#128188; Registered by:</strong> {admin_name or 'Admin'}</p>
          {'<p style="color:#4a5568;font-size:.82rem;margin:.3rem 0 0;"><strong>Admin Email:</strong> ' + admin_email + '</p>' if admin_email else ''}
          <p style="color:#888;font-size:.78rem;margin:.3rem 0 0;">For any changes to your Department or Designation, please contact your admin.</p>
        </div>"""
    return f"""
    <div style="font-family:'Segoe UI',Arial,sans-serif;max-width:560px;margin:0 auto;background:#fff;border-radius:16px;overflow:hidden;box-shadow:0 4px 24px rgba(0,0,0,.1);">
      <div style="background:{gradient};padding:2rem;text-align:center;color:#fff;">
        <div style="width:60px;height:60px;background:rgba(255,255,255,.2);border-radius:50%;margin:0 auto 1rem;display:flex;align-items:center;justify-content:center;font-size:1.6rem;">&#128100;</div>
        <h2 style="margin:0;font-size:1.4rem;">Your Account is Ready!</h2>
        <p style="opacity:.85;margin:.4rem 0 0;font-size:.9rem;">Smart Attendance System — {role} Portal</p>
      </div>
      <div style="padding:2rem;">
        <p style="color:#555;font-size:.95rem;">Hello <strong>{full_name}</strong>,</p>
        <p style="color:#555;font-size:.95rem;">An account has been created for you on the Smart Attendance System. Here are your complete account details:</p>
        <div style="background:#f8f9ff;border:1px solid #e8ecff;border-radius:12px;padding:1.2rem;margin:1.2rem 0;">
          <table style="width:100%;border-collapse:collapse;">
            <tr><td style="padding:8px 12px;color:#888;font-size:.85rem;font-weight:600;white-space:nowrap;">Full Name</td>
                <td style="padding:8px 12px;font-size:.95rem;font-weight:700;color:#333;">{full_name}</td></tr>
            <tr><td style="padding:8px 12px;color:#888;font-size:.85rem;font-weight:600;border-top:1px solid #eef;white-space:nowrap;">User ID</td>
                <td style="padding:8px 12px;font-size:1.1rem;font-weight:800;color:#333;letter-spacing:2px;border-top:1px solid #eef;">{user_id}</td></tr>
            <tr><td style="padding:8px 12px;color:#888;font-size:.85rem;font-weight:600;border-top:1px solid #eef;white-space:nowrap;">User PIN</td>
                <td style="padding:8px 12px;font-size:1.3rem;font-weight:800;color:#333;letter-spacing:6px;border-top:1px solid #eef;">{user_pin}</td></tr>
            {extra_rows}
          </table>
        </div>
        {admin_section}
        <p style="color:#e74c3c;font-size:.82rem;margin:.8rem 0;">&#9888; Keep your PIN confidential. Do not share it with anyone.</p>
        <div style="background:#fff8e1;border:1px solid #ffe082;border-radius:10px;padding:.8rem 1rem;margin:1rem 0;">
          <p style="color:#f57f17;font-size:.82rem;margin:0;"><strong>How to login:</strong> Go to the Teacher/Staff Portal &rarr; Enter your <strong>User ID</strong> and <strong>User PIN</strong> to sign in.</p>
        </div>
      </div>
      <div style="background:#f8f8ff;padding:1rem;text-align:center;font-size:.75rem;color:#aaa;border-top:1px solid #eee;">Smart Attendance System &copy; 2025</div>
    </div>
    """


# Database setup
def get_db_connection():
    conn = sqlite3.connect("attendance.db")
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    with open("schema.sql") as f:
        conn.executescript(f.read())
    conn.commit()
    # Ensure new columns exist on existing DBs (migrations)
    _migrations = [
        ("classes", "department", "ALTER TABLE classes ADD COLUMN department TEXT"),
        ("attendance_records", "attendance_type", "ALTER TABLE attendance_records ADD COLUMN attendance_type TEXT NOT NULL DEFAULT 'gate'"),
        ("attendance_records", "admin_id", "ALTER TABLE attendance_records ADD COLUMN admin_id INTEGER"),
        ("admins", "college_name", "ALTER TABLE admins ADD COLUMN college_name TEXT DEFAULT ''"),
        ("admins", "phone", "ALTER TABLE admins ADD COLUMN phone TEXT DEFAULT ''"),
        ("admins", "profile_image", "ALTER TABLE admins ADD COLUMN profile_image TEXT DEFAULT ''"),
        ("admins", "bio", "ALTER TABLE admins ADD COLUMN bio TEXT DEFAULT ''"),
        ("admins", "designation", "ALTER TABLE admins ADD COLUMN designation TEXT DEFAULT ''"),
        ("teachers", "profile_image", "ALTER TABLE teachers ADD COLUMN profile_image TEXT DEFAULT ''"),
        ("teachers", "phone", "ALTER TABLE teachers ADD COLUMN phone TEXT DEFAULT ''"),
        ("teachers", "department", "ALTER TABLE teachers ADD COLUMN department TEXT DEFAULT ''"),
        ("teachers", "bio", "ALTER TABLE teachers ADD COLUMN bio TEXT DEFAULT ''"),
        ("teachers", "designation", "ALTER TABLE teachers ADD COLUMN designation TEXT DEFAULT ''"),
        ("users", "registered_by", "ALTER TABLE users ADD COLUMN registered_by TEXT DEFAULT ''"),
        ("users", "registered_by_role", "ALTER TABLE users ADD COLUMN registered_by_role TEXT DEFAULT ''"),
        ("admins", "institution_type", "ALTER TABLE admins ADD COLUMN institution_type TEXT DEFAULT 'college'"),
        ("email_verifications", "institution_type", "ALTER TABLE email_verifications ADD COLUMN institution_type TEXT DEFAULT 'college'"),
        ("users", "is_active", "ALTER TABLE users ADD COLUMN is_active INTEGER DEFAULT 1"),
        ("teachers", "user_pin", "ALTER TABLE teachers ADD COLUMN user_pin TEXT DEFAULT ''"),
        ("admins", "user_pin", "ALTER TABLE admins ADD COLUMN user_pin TEXT DEFAULT ''"),
        ("users", "registered_by_admin_id", "ALTER TABLE users ADD COLUMN registered_by_admin_id INTEGER DEFAULT 0"),
        ("teachers", "registered_by_admin_id", "ALTER TABLE teachers ADD COLUMN registered_by_admin_id INTEGER DEFAULT 0"),
        ("password_resets", "user_type", "ALTER TABLE password_resets ADD COLUMN user_type TEXT DEFAULT 'teacher'"),
    ]
    for table, col, sql in _migrations:
        try:
            conn.execute(f"SELECT {col} FROM {table} LIMIT 1")
        except sqlite3.OperationalError:
            try:
                conn.execute(sql)
                conn.commit()
            except sqlite3.OperationalError:
                pass
    # Make teacher_id nullable in existing attendance_records (for admin-only records)
    conn.close()


# Initialize database (always run to ensure all tables exist)
init_db()


# Create email verification & password reset tables
def init_auth_tables():
    conn = get_db_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS email_verifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL,
            username TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            full_name TEXT NOT NULL,
            otp_code TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP NOT NULL,
            verified INTEGER DEFAULT 0,
            user_type TEXT NOT NULL DEFAULT 'teacher',
            college_name TEXT DEFAULT '',
            phone TEXT DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS password_resets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            teacher_id INTEGER NOT NULL,
            token TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP NOT NULL,
            used INTEGER DEFAULT 0,
            FOREIGN KEY (teacher_id) REFERENCES teachers (id)
        );
    """)
    # Safely add columns if they don't exist
    for col, default in [("is_verified", "1")]:
        try:
            conn.execute(f"ALTER TABLE teachers ADD COLUMN {col} INTEGER DEFAULT {default}")
            conn.commit()
        except sqlite3.OperationalError:
            pass
    for col in ["user_type", "college_name", "phone"]:
        try:
            conn.execute(f"ALTER TABLE email_verifications ADD COLUMN {col} TEXT DEFAULT ''")
            conn.commit()
        except sqlite3.OperationalError:
            pass
    # Add user_id column to teachers and admins tables
    for table in ["teachers", "admins"]:
        try:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN user_id TEXT DEFAULT ''")
            conn.commit()
        except sqlite3.OperationalError:
            pass
        # Backfill existing rows that have no user_id
        rows = conn.execute(f"SELECT id FROM {table} WHERE user_id IS NULL OR user_id = ''").fetchall()
        prefix = "TCH" if table == "teachers" else "ADM"
        for row in rows:
            uid = f"{prefix}-{row['id']:04d}"
            conn.execute(f"UPDATE {table} SET user_id = ? WHERE id = ?", (uid, row['id']))
        if rows:
            conn.commit()
    conn.close()


def generate_user_id(prefix, conn, table):
    """Auto-generate a unique user ID like TCH-0001, ADM-0001"""
    result = conn.execute(
        f"SELECT user_id FROM {table} WHERE user_id LIKE ? ORDER BY id DESC LIMIT 1",
        (f"{prefix}-%",)
    ).fetchone()
    if result and result['user_id']:
        try:
            last_num = int(result['user_id'].split('-')[1])
            return f"{prefix}-{last_num + 1:04d}"
        except (ValueError, IndexError):
            pass
    # Fallback: count rows
    count = conn.execute(f"SELECT COUNT(*) as c FROM {table}").fetchone()['c']
    return f"{prefix}-{count + 1:04d}"


init_auth_tables()


def speak(message):
    speaker = Dispatch("SAPI.SpVoice")
    speaker.Speak(message)


# Initialize Haar Cascade
# First try to use the local file
local_cascade_path = (
    r"C:\Users\MANISH SHARMA\OneDrive\Desktop\Smart Attendence System\haarcascade_frontalface_default.xml"
)
if os.path.exists(local_cascade_path):
    print(f"Using local cascade file: {local_cascade_path}")
    facedetect = cv2.CascadeClassifier(local_cascade_path)
else:
    # Fallback to OpenCV's built-in cascade file
    print("Local cascade file not found, using OpenCV's built-in cascade")
    facedetect = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

# Check if the classifier loaded successfully
if facedetect.empty():
    print("ERROR: Could not load the face cascade classifier!")
    # Try alternate locations
    alt_path = "haarcascade_frontalface_default.xml"  # Try without space
    if os.path.exists(alt_path):
        print(f"Using alternate cascade file: {alt_path}")
        facedetect = cv2.CascadeClassifier(alt_path)

# Create directory for saving data
if not os.path.exists("data"):
    os.makedirs("data")


# Function to capture new faces
def capture_faces(name):
    # Make sure data directory exists
    if not os.path.exists("data"):
        os.makedirs("data")
        print("Created data directory")

    # Initialize the camera - try DirectShow first, then default
    video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not video.isOpened():
        print("DirectShow backend failed, trying default backend")
        video = cv2.VideoCapture(0)

    # Set basic camera properties
    video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Check if camera opened successfully
    if not video.isOpened():
        print("Error: Could not open camera with any backend")
        return "Error: Camera not accessible. Please check your webcam connection."

    # Display status message
    print(f"Starting face capture for {name}")

    faces_data = []
    required_samples = 50  # Number of face captures needed

    # Simpler frame skipping for diversity
    capture_delay = 2  # Capture every 2 frames
    frame_count = 0

    # Simple countdown before starting
    countdown = 3
    countdown_time = time.time()
    capturing_started = False
    capture_timeout = 30  # seconds timeout for face detection
    capture_start_time = None

    while True:
        # Read a frame from the video
        ret, frame = video.read()
        if not ret:
            print("Error accessing camera.")
            break

        # Get current time
        current_time = time.time()

        # Make a copy for display
        display_frame = frame.copy()

        # Show countdown before starting capture
        if not capturing_started:
            # Draw a background for text
            cv2.rectangle(display_frame, (0, 0), (640, 100), (40, 40, 40), -1)

            # Update countdown every second
            if current_time - countdown_time >= 1:
                countdown -= 1
                countdown_time = current_time
                if countdown <= 0:
                    capturing_started = True
                    capture_start_time = current_time

            # Draw countdown message
            cv2.putText(
                display_frame,
                f"Get ready! Starting in {countdown}...",
                (40, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

            # Show the frame
            cv2.imshow("Face Capture", display_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            continue

        # Process frame for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces - use more reliable parameters
        faces = facedetect.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20)
        )

        # Draw simple UI - progress info
        cv2.rectangle(display_frame, (0, 0), (640, 50), (40, 40, 40), -1)
        progress_text = f"Capturing: {len(faces_data)}/{required_samples} images"
        cv2.putText(
            display_frame,
            progress_text,
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        # Found a face?
        if len(faces) > 0:
            # Get the largest face
            x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])

            # Draw rectangle around face
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Capture face every few frames
            if len(faces_data) < required_samples and frame_count % capture_delay == 0:
                # Get face with small padding
                padding = int(0.1 * w)  # 10% padding
                y1 = max(0, y - padding)
                y2 = min(frame.shape[0], y + h + padding)
                x1 = max(0, x - padding)
                x2 = min(frame.shape[1], x + w + padding)

                # Crop and resize the face
                try:
                    crop_img = frame[y1:y2, x1:x2]
                    if crop_img.size > 0:  # Make sure we have a valid image
                        resized_img = cv2.resize(crop_img, (50, 50))
                        faces_data.append(resized_img)

                        # Simple visual feedback - green border means captured
                        cv2.rectangle(
                            display_frame, (x, y), (x + w, y + h), (0, 255, 0), 3
                        )
                except Exception as e:
                    print(f"Error cropping face: {e}")
        else:
            # No face detected
            cv2.putText(
                display_frame,
                "No face detected",
                (180, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

        # Increment frame counter
        frame_count += 1

        # Show the frame
        cv2.imshow("Face Capture", display_frame)

        # Abort if no face detected within timeout
        if capture_start_time and time.time() - capture_start_time > capture_timeout:
            print(
                f"Timeout: No face detected within {capture_timeout} seconds, aborting capture."
            )
            video.release()
            cv2.destroyAllWindows()
            return "Error: Could not detect face. Please ensure proper lighting and try again."

        # Check for user quit or completion
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or len(faces_data) >= required_samples:
            break

    # Clean up
    video.release()
    cv2.destroyAllWindows()

    # Make sure we have enough samples
    if len(faces_data) < required_samples:
        print(
            f"Warning: Only captured {len(faces_data)} samples out of required {required_samples}"
        )
        # If we have at least 1 sample, we can proceed, otherwise return error
        if len(faces_data) < 1:
            return "Error: Could not capture any face samples. Please try again with better lighting."
        # Adjust required_samples to what we have
        required_samples = len(faces_data)

    print(f"Successfully captured {len(faces_data)} face samples")

    # Save the data - ensure consistent dimensions (50x50 = 7500 pixels per face for RGB)
    faces_data = np.asarray(faces_data)
    # Check for empty array
    if len(faces_data) == 0:
        return "Error: No face samples captured. Please try again with better lighting."

    # Print shape before reshaping to debug
    print(f"Face data shape before reshaping: {faces_data.shape}")
    # Make sure all faces have the right dimensions - 50x50 RGB images = 7500 elements per face
    try:
        # First ensure all samples have the same shape by resizing if needed
        for i in range(len(faces_data)):
            if faces_data[i].shape != (50, 50, 3):
                faces_data[i] = cv2.resize(faces_data[i], (50, 50))

        # Then reshape to a 2D array for the classifier
        faces_data = faces_data.reshape(len(faces_data), 50 * 50 * 3)
        print(f"Face data successfully reshaped to: {faces_data.shape}")
    except Exception as e:
        print(f"Error reshaping face data: {str(e)}")
        # Create a fresh array with correct dimensions as fallback
        faces_data = np.zeros((len(faces_data), 50 * 50 * 3))
        for i in range(len(faces_data)):
            flat = cv2.resize(faces_data[i], (50, 50)).flatten()
            faces_data[i] = flat if flat.size == 50 * 50 * 3 else np.zeros(50 * 50 * 3)
        print(f"Created fallback face data with shape: {faces_data.shape}")

    # Create data directory if it doesn't exist
    if not os.path.exists("data"):
        os.makedirs("data")
        print("Created data directory")

    try:
        if not os.path.exists("data/names.pkl"):
            names = [name] * len(faces_data)
            with open("data/names.pkl", "wb") as f:
                pickle.dump(names, f)
            print("Created new names file")
        else:
            with open("data/names.pkl", "rb") as f:
                names = pickle.load(f)
            names += [name] * len(faces_data)
            with open("data/names.pkl", "wb") as f:
                pickle.dump(names, f)
            print(f"Updated existing names file, now contains {len(names)} samples")

        # Save the faces data
        if not os.path.exists("data/faces_data.pkl"):
            with open("data/faces_data.pkl", "wb") as f:
                pickle.dump(faces_data, f)
            print("Created new faces data file")
        else:
            try:
                with open("data/faces_data.pkl", "rb") as f:
                    faces = pickle.load(f)

                # Print shapes for debugging
                print(f"Existing faces data shape: {faces.shape}")
                print(f"New faces data shape: {faces_data.shape}")

                # Check and fix dimensions if needed
                if faces.size == 0:
                    print("Existing faces data is empty, using new data only")
                    faces = faces_data
                elif faces.shape[1] != faces_data.shape[1]:
                    print(
                        f"Dimension mismatch: existing={faces.shape[1]}, new={faces_data.shape[1]}"
                    )
                    # Try to reshape old data to match new data dimensions
                    try:
                        # If old data has wrong dimensions, we'll recreate it with the right dimensions
                        # We'll store the old data temporarily for names matching
                        with open("data/names.pkl", "rb") as f:
                            old_names = pickle.load(f)

                        # Create new faces array with right dimensions
                        print("Recreating faces data with consistent dimensions")
                        # Just use the new data and discard old data with wrong dimensions
                        # This is safer than trying to reshape incompatible data
                        faces = faces_data

                        # Update names list to match the new face data size
                        # This ensures names and faces stay aligned
                        with open("data/names.pkl", "wb") as f:
                            pickle.dump([name] * len(faces_data), f)
                        print("Names file has been reset to match new face data")
                    except Exception as e:
                        print(f"Error fixing dimension mismatch: {str(e)}")
                        faces = faces_data
                else:
                    # Append only if dimensions match
                    faces = np.append(faces, faces_data, axis=0)
                    print(
                        f"Successfully appended new face data, new shape: {faces.shape}"
                    )
            except Exception as e:
                print(
                    f"Error loading existing face data: {str(e)}. Creating new dataset."
                )
                faces = faces_data
            with open("data/faces_data.pkl", "wb") as f:
                pickle.dump(faces, f)
            print(f"Updated faces data file, now contains {faces.shape[0]} samples")
    except Exception as e:
        print(f"Error saving face data: {str(e)}")
        return f"Error saving data: {str(e)}"

    return "Dataset created successfully!"


# Function to start face recognition and attendance
def start_recognition(class_id=None):
    try:
        # Check if data directory exists
        if not os.path.exists("data"):
            print("Error: Data directory not found")
            os.makedirs("data")
            print("Created data directory, but no facial data exists yet")
            return "Please register at least one face before starting recognition"

        # Check if required files exist with detailed feedback
        missing_files = []
        if not os.path.exists("data/names.pkl"):
            missing_files.append("names.pkl")
        if not os.path.exists("data/faces_data.pkl"):
            missing_files.append("faces_data.pkl")

        if missing_files:
            print(f"Error: Required data files not found: {', '.join(missing_files)}")
            return "Please register at least one face before starting recognition"

        # Load Data for Recognition
        try:
            with open("data/names.pkl", "rb") as f:
                names = pickle.load(f)
            print(f"Successfully loaded names.pkl with {len(names)} names")
            if len(names) == 0:
                print("Error: names list is empty")
                return "No registered users found. Please register at least one face before starting recognition"
        except Exception as e:
            print(f"Error loading names.pkl: {str(e)}")
            return "Error loading face data. Please try registering your face again"

        try:
            with open("data/faces_data.pkl", "rb") as f:
                faces_data = pickle.load(f)
            print(f"Successfully loaded faces_data.pkl with shape {faces_data.shape}")
            if faces_data.size == 0:
                print("Error: faces_data array is empty")
                return "Error: No face data found"
        except Exception as e:
            print(f"Error loading faces_data.pkl: {str(e)}")
            return "Error: Failed to load faces data"

        try:
            faces_data = faces_data.reshape(faces_data.shape[0], -1)
            print(f"Successfully reshaped faces_data to {faces_data.shape}")
            print(f"Number of face samples: {faces_data.shape[0]}")
            print(f"Number of names: {len(names)}")

            # Fix inconsistency between faces and names instead of returning error
            if faces_data.shape[0] != len(names):
                print(
                    f"Warning: Mismatch between faces_data.shape[0] ({faces_data.shape[0]}) and len(names) ({len(names)})"
                )
                print("Attempting to fix the inconsistency automatically...")

                try:
                    # Case 1: More faces than names - truncate face data to match names
                    if faces_data.shape[0] > len(names):
                        print("More faces than names, truncating face data")
                        faces_data = faces_data[: len(names)]
                    # Case 2: More names than faces - duplicate the last name for each remaining face
                    else:
                        print("More names than faces, adjusting names list")
                        # Take a subset of names to match number of faces
                        names = names[: faces_data.shape[0]]

                    # Save the corrected data
                    with open("data/names.pkl", "wb") as f:
                        pickle.dump(names, f)
                    with open("data/faces_data.pkl", "wb") as f:
                        pickle.dump(faces_data, f)
                    print("Successfully fixed data inconsistency")
                except Exception as e:
                    print(f"Error fixing data inconsistency: {str(e)}")
                    return "Error: Failed to fix inconsistent face data. Please register your face again."
        except Exception as e:
            print(f"Error reshaping faces_data: {str(e)}")
            return "Error: Failed to process faces data"

        # Initialize KNN Classifier with robust error handling
        try:
            if faces_data.shape[0] < 1:
                print("Error: No face samples for KNN classifier")
                return "Error: No face samples for recognition"

            # Make sure names and faces_data have same length
            if len(names) != faces_data.shape[0]:
                print("Warning: Length mismatch before training, fixing...")
                # Use the minimum length between names and faces_data
                min_len = min(len(names), faces_data.shape[0])
                names = names[:min_len]
                faces_data = faces_data[:min_len]

            # Use a more appropriate number of neighbors
            n_neighbors = min(3, faces_data.shape[0])
            print(f"Using {n_neighbors} neighbors for KNN classification")

            # Verify data is suitable for training (no NaNs or infs)
            if np.isnan(faces_data).any() or np.isinf(faces_data).any():
                print("Warning: Data contains NaN or Inf values, cleaning data...")
                # Replace NaN and Inf values with zeros
                faces_data = np.nan_to_num(faces_data)

            knn = KNeighborsClassifier(n_neighbors=n_neighbors)
            knn.fit(faces_data, names)
            print("Successfully trained KNN classifier")
        except Exception as e:
            print(f"Error training classifier: {str(e)}")
            return f"Error: Could not train recognition model: {str(e)}"

        # Video Capture for Recognition - use the same settings as in capture
        video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not video.isOpened():
            print("DirectShow backend failed, trying default backend")
            video = cv2.VideoCapture(0)

        # Set basic camera properties to match capture function
        video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not video.isOpened():
            print("Error: Could not open video capture device with any backend")
            return "Error: Camera not accessible"

        # Create a simple colored background instead of loading an image
        # This eliminates any dependency on external image files
        imgBackground = np.zeros((480, 640, 3), dtype=np.uint8)
        # Fill with a pleasant blue gradient background
        for y in range(480):
            for x in range(640):
                # Create a gradient from dark blue to light blue
                blue_value = int(180 + (y / 480) * 75)  # 180-255 range for blue
                imgBackground[y, x] = (100, 50, blue_value)  # BGR format

        print("Using generated background instead of image file")

        COL_NAMES = ["NAME", "DATE", "TIME"]

        attendance_dir = r"C:\Users\MANISH SHARMA\OneDrive\Desktop\Smart Attendence System\Attendance"
        os.makedirs(attendance_dir, exist_ok=True)

        # Instead of using a fixed recognition threshold, we'll use confidence level
        min_confidence_for_attendance = (
            70  # Minimum confidence percentage to record attendance
        )

        attendance_list = []  # List to store multiple attendance records
        attendance_recorded = (
            {}
        )  # Dictionary to track when attendance was last recorded

        while True:
            ret, frame = video.read()
            if not ret:
                print("Error accessing camera.")
                break

            # Process frame with improved parameters for better recognition
            # Use the full frame for better quality
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Apply histogram equalization to improve contrast
            gray = cv2.equalizeHist(gray)

            # Use more lenient detection parameters
            faces = facedetect.detectMultiScale(
                gray,
                scaleFactor=1.05,  # More sensitive scale factor
                minNeighbors=3,  # Balanced for accuracy and detection ease
                minSize=(20, 20),  # Minimum face size - same as in capture
            )

            # Display guidance text with instructions
            cv2.rectangle(frame, (0, 0), (640, 80), (40, 40, 40), -1)
            cv2.putText(
                frame,
                "Position your face in front of the camera",
                (20, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                "Window will close automatically after attendance is recorded",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
            )
            cv2.putText(
                frame,
                "Press 'q' to quit without recording, 'm' for manual attendance",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
            )

            for x, y, w, h in faces:
                crop_img = frame[y : y + h, x : x + w, :]
                # Make sure we have a valid image
                if crop_img.size == 0 or crop_img is None:
                    continue

                # Apply same preprocessing as during capture
                try:
                    resized_img = cv2.resize(crop_img, (50, 50))
                    flattened_img = resized_img.flatten().reshape(1, -1)

                    # Normalize the image for better recognition
                    norm_img = cv2.normalize(
                        flattened_img, None, 0, 255, cv2.NORM_MINMAX
                    )
                except Exception as e:
                    print(f"Error processing face image: {e}")
                    continue

                # Try to predict the name in real-time
                try:
                    # Get prediction directly - use voting from multiple neighbors
                    prediction = knn.predict(norm_img)[0]

                    # Get probabilities or distance from neighbors
                    n_neighbors = min(5, faces_data.shape[0])
                    distances, indices = knn.kneighbors(
                        norm_img, n_neighbors=n_neighbors
                    )

                    # Calculate confidence based on neighbor votes instead of distance
                    neighbors = [names[idx] for idx in indices[0]]
                    votes = {}
                    for neighbor in neighbors:
                        if neighbor in votes:
                            votes[neighbor] += 1
                        else:
                            votes[neighbor] = 1

                    # Get vote count for the predicted name
                    vote_count = votes.get(prediction, 0)
                    confidence = (vote_count / n_neighbors) * 100

                    # Print recognition info
                    print(
                        f"Predicted: {prediction}, Confidence: {confidence:.0f}%, Votes: {votes}"
                    )

                    # Determine if face is recognized with sufficient confidence
                    if confidence < 40:  # Very low confidence
                        name = "Unknown"
                        color = (0, 0, 255)  # Red for unknown
                    else:
                        # Get the predicted name with confidence
                        name = prediction
                        name = f"{name} ({confidence:.0f}%)"
                        color = (0, 255, 0)  # Green for recognized

                        # Automatically record attendance if confidence is high enough
                        if confidence >= min_confidence_for_attendance:
                            # Check if this person's attendance was already recorded recently
                            current_time = time.time()
                            # Only record if we haven't recorded in the last 60 seconds
                            if (
                                prediction not in attendance_recorded
                                or (current_time - attendance_recorded[prediction]) > 60
                            ):
                                ts = time.time()
                                record_date = datetime.fromtimestamp(ts).strftime(
                                    "%d-%m-%Y"
                                )
                                timestamp = datetime.fromtimestamp(ts).strftime(
                                    "%H:%M:%S"
                                )
                                attendance_list.append(
                                    [prediction, record_date, timestamp]
                                )
                                speak(f"Attendance recorded for {prediction}")
                                attendance_recorded[prediction] = current_time

                                # Show confirmation message
                                cv2.rectangle(
                                    frame, (0, 0), (640, 100), (0, 150, 0), -1
                                )
                                cv2.putText(
                                    frame,
                                    f"Attendance recorded for {prediction}",
                                    (20, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1,
                                    (255, 255, 255),
                                    2,
                                )
                                cv2.putText(
                                    frame,
                                    "Saving and closing in 2 seconds...",
                                    (20, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7,
                                    (255, 255, 255),
                                    1,
                                )
                                cv2.imshow("Face Recognition", frame)
                                cv2.waitKey(2000)  # Show confirmation for 2 seconds

                                # Save attendance immediately
                                save_attendance_now(
                                    attendance_list, attendance_dir, COL_NAMES, class_id
                                )

                                # Close video and window
                                video.release()
                                cv2.destroyAllWindows()
                                return "Attendance recorded successfully!"
                except Exception as e:
                    name = "Error"
                    color = (0, 165, 255)  # Orange for error
                    print(f"Error during prediction: {str(e)}")

                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                # Create dark background for text
                cv2.rectangle(frame, (x, y - 30), (x + w, y), color, -1)

                # Display name on rectangle
                cv2.putText(
                    frame,
                    name,
                    (x + 5, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                )

            cv2.imshow("Face Recognition", frame)

            key = cv2.waitKey(1)
            if key == ord("o"):  # Press 'o' to mark attendance
                for x, y, w, h in faces:
                    crop_img = frame[y : y + h, x : x + w, :]
                    resized_img = (
                        cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
                    )

                    # Apply preprocessing to the face image for better recognition
                    # Normalize the image to reduce lighting effects
                    norm_img = cv2.normalize(resized_img, None, 0, 255, cv2.NORM_MINMAX)

                    # Try to predict with error handling - simpler approach for manual attendance
                    try:
                        # Just get the prediction directly
                        prediction = knn.predict(norm_img)[0]
                        name = prediction
                        color = (0, 255, 0)  # Green for recognized
                    except Exception as e:
                        print(f"Error during prediction: {str(e)}")
                        name = "Error"
                        color = (0, 165, 255)  # Orange for error

                    # Draw bounding box and label
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
                    cv2.putText(
                        frame,
                        str(name),
                        (x, y - 15),
                        cv2.FONT_HERSHEY_COMPLEX,
                        1,
                        (255, 255, 255),
                        1,
                    )

                    ts = time.time()
                    record_date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
                    timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
                    attendance_list.append([name, record_date, timestamp])
                    speak(f"Attendance recorded for {name}")

            elif key == ord("m"):  # Press 'm' to manually record attendance
                for x, y, w, h in faces:
                    crop_img = frame[y : y + h, x : x + w, :]
                    if crop_img.size == 0 or crop_img is None:
                        continue

                    try:
                        resized_img = cv2.resize(crop_img, (50, 50))
                        flattened_img = resized_img.flatten().reshape(1, -1)
                        norm_img = cv2.normalize(
                            flattened_img, None, 0, 255, cv2.NORM_MINMAX
                        )

                        # Get prediction
                        prediction = knn.predict(norm_img)[0]

                        # Manual recording always uses the predicted name
                        ts = time.time()
                        record_date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
                        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
                        attendance_list.append([prediction, record_date, timestamp])
                        speak(f"Attendance manually recorded for {prediction}")

                        # Update screen with confirmation
                        cv2.rectangle(frame, (0, 0), (640, 40), (0, 150, 0), -1)
                        cv2.putText(
                            frame,
                            f"Attendance recorded for {prediction}",
                            (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 255),
                            2,
                        )
                        cv2.imshow("Face Recognition", frame)
                        cv2.waitKey(1000)  # Show the confirmation for 1 second
                    except Exception as e:
                        print(f"Error recording manual attendance: {e}")

            elif key == ord("q"):  # Press 'q' to quit
                break

        # If we reach here, the user manually quit without recording attendance
        video.release()
        cv2.destroyAllWindows()

        # Save attendance if any was recorded but not saved
        if attendance_list:
            save_attendance_now(attendance_list, attendance_dir, COL_NAMES)

        return "Recognition complete and attendance taken!"

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return f"Error: An unexpected error occurred - {str(e)}"


# Function to save attendance immediately
def save_attendance_now(attendance_list, attendance_dir, COL_NAMES, class_id=None):
    if not attendance_list:
        return

    # Create attendance file with current date
    current_date = datetime.now().strftime("%d-%m-%Y")
    attendance_file = os.path.join(attendance_dir, f"Attendance_{current_date}.csv")
    attendance_exists = os.path.exists(attendance_file)

    # Write all collected attendance records
    with open(attendance_file, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not attendance_exists:
            writer.writerow(COL_NAMES)
        # Write all records
        writer.writerows(attendance_list)

    # Store attendance in database regardless of teacher login
    for record in attendance_list:
        student_name, date_str, time_str = record
        if student_name != "Unknown" and student_name != "Error":
            # Pass the class_id to the store_attendance_in_db function
            store_attendance_in_db(student_name, date_str, time_str, class_id)
    speak("Attendance has been saved")
    return "Attendance saved successfully!"


# User Management functions
def get_all_users():
    if os.path.exists("data/names.pkl"):
        with open("data/names.pkl", "rb") as f:
            names = pickle.load(f)
        return list(set(names))  # Return unique names
    return []


def delete_user(username):
    if os.path.exists("data/names.pkl"):
        with open("data/names.pkl", "rb") as f:
            names = pickle.load(f)
        # Remove all instances of the username
        names = [name for name in names if name != username]
        with open("data/names.pkl", "wb") as f:
            pickle.dump(names, f)
        return True
    return False


# Settings functions
def save_settings(settings_dict):
    with open("data/settings.pkl", "wb") as f:
        pickle.dump(settings_dict, f)


def load_settings():
    defaults = {
        "camera_index": 0,
        "required_samples": 50,
        "attendance_threshold": 0.6,
        "min_confidence": 50,
        "auto_save_attendance": True,
        "recognition_interval": 2000,
        "departments": ["Computer Science", "IT", "Electronics", "Mechanical", "Civil", "Other"],
        "roles": ["Student", "Teacher", "Staff", "Admin"],
        "duplicate_check": True,
        "notification_sound": True,
    }
    if os.path.exists("data/settings.pkl"):
        with open("data/settings.pkl", "rb") as f:
            saved = pickle.load(f)
        # Merge saved with defaults (so new keys get default values)
        defaults.update(saved)
    return defaults


# Export functions
def export_attendance_csv():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"attendance_export_{timestamp}.csv"
    export_path = os.path.join("data", filename)

    if os.path.exists("data/attendance.csv"):
        with open("data/attendance.csv", "r") as source:
            with open(export_path, "w", newline="") as target:
                reader = csv.reader(source)
                writer = csv.writer(target)
                for row in reader:
                    writer.writerow(row)
        return filename
    return None


# Routes
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/instructions")
def instructions():
    return render_template("instructions.html")


@app.route("/capture", methods=["GET"])
@app.route("/admin/capture", methods=["GET"], endpoint="admin_capture")
@app.route("/teacher/capture", methods=["GET"], endpoint="teacher_capture")
@any_login_required
def capture():
    classes = []
    if "teacher_id" in session:
        conn = get_db_connection()
        classes = conn.execute(
            "SELECT id, name FROM classes WHERE teacher_id = ?", (session["teacher_id"],)
        ).fetchall()
        conn.close()
    return render_template("capture.html", classes=classes)


@app.route("/api/upload_capture", methods=["POST"])
def upload_capture():
    """
    API endpoint for web-based face capture.
    Receives base64 encoded images and saves them for training.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data received"}), 400

        username = (data.get("username") or "").strip()
        email = (data.get("email") or "").strip()
        user_id = (data.get("userId") or "").strip()
        department = (data.get("department") or "").strip()
        phone = (data.get("phone") or "").strip()
        role = (data.get("role") or "student").strip()
        notes = (data.get("notes") or "").strip()
        user_pin_plain = (data.get("user_pin") or "").strip()
        images = data.get("images", [])

        # ── Validate required fields ──
        if not username:
            return jsonify({"error": "Name is required"}), 400
        if not all(c.isalpha() or c.isspace() for c in username):
            return jsonify({"error": "Name must contain only letters and spaces"}), 400
        if not email or "@" not in email:
            return jsonify({"error": "Valid email is required"}), 400
        if not user_id:
            return jsonify({"error": "User ID is required"}), 400
        if not department:
            return jsonify({"error": "Department is required"}), 400
        if not images or len(images) < 5:
            return jsonify({"error": f"At least 5 images required (received {len(images)})"}), 400

        # Cap at 100 images
        if len(images) > 100:
            images = images[:100]

        # Create data directory
        os.makedirs("data", exist_ok=True)

        # ── Process images: detect & crop faces ──
        faces_data = []
        for idx, image_data in enumerate(images):
            try:
                if "," in image_data:
                    image_data = image_data.split(",")[1]
                image_bytes = base64.b64decode(image_data)
                nparr = np.frombuffer(image_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.equalizeHist(gray)

                # Try multiple cascade parameters for robustness
                faces = facedetect.detectMultiScale(gray, 1.3, 5)
                if len(faces) == 0:
                    faces = facedetect.detectMultiScale(gray, 1.1, 3, minSize=(30, 30))

                if len(faces) > 0:
                    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
                    crop_img = frame[y : y + h, x : x + w]
                    resized_img = cv2.resize(crop_img, (50, 50))
                    faces_data.append(resized_img)
                else:
                    # Fallback: use center-crop of the frame as a face
                    h, w = frame.shape[:2]
                    sz = min(h, w)
                    y0, x0 = (h - sz) // 2, (w - sz) // 2
                    crop = frame[y0 : y0 + sz, x0 : x0 + sz]
                    resized_img = cv2.resize(crop, (50, 50))
                    faces_data.append(resized_img)
            except Exception as e:
                print(f"Error processing image {idx}: {e}")
                continue

        if len(faces_data) < 5:
            return jsonify({"error": f"Only {len(faces_data)} usable images. Please ensure good lighting and face visibility."}), 400

        new_faces = np.array(faces_data)
        new_count = len(new_faces)

        # ── Merge with existing data ──
        try:
            if os.path.exists("data/faces_data.pkl") and os.path.exists("data/names.pkl"):
                with open("data/faces_data.pkl", "rb") as f:
                    existing_faces = pickle.load(f)
                with open("data/names.pkl", "rb") as f:
                    existing_names = pickle.load(f)
                all_faces = np.append(existing_faces, new_faces, axis=0)
                all_names = list(existing_names) + [username] * new_count
            else:
                all_faces = new_faces
                all_names = [username] * new_count
        except Exception as e:
            print(f"Error loading existing data, starting fresh: {e}")
            all_faces = new_faces
            all_names = [username] * new_count

        # ── Save data ──
        with open("data/faces_data.pkl", "wb") as f:
            pickle.dump(all_faces, f)
        with open("data/names.pkl", "wb") as f:
            pickle.dump(all_names, f)

        # ── Train KNN model ──
        n_neighbors = min(5, len(all_faces))
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(all_faces.reshape(all_faces.shape[0], -1), all_names)
        with open("data/face_recognizer.pkl", "wb") as f:
            pickle.dump(knn, f)

        # ── Save user to database ──
        conn = get_db_connection()
        try:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    email TEXT NOT NULL,
                    user_id TEXT UNIQUE NOT NULL,
                    department TEXT NOT NULL,
                    phone TEXT,
                    role TEXT NOT NULL,
                    notes TEXT,
                    created_at TEXT NOT NULL,
                    registered_by TEXT DEFAULT '',
                    registered_by_role TEXT DEFAULT '',
                    registered_by_admin_id INTEGER DEFAULT 0,
                    is_active INTEGER DEFAULT 1
                )"""
            )
            # Determine who registered this user
            reg_by = ''
            reg_by_role = ''
            reg_admin_id = 0
            if 'admin_id' in session:
                admin_row = conn.execute('SELECT full_name FROM admins WHERE id = ?', (session['admin_id'],)).fetchone()
                reg_by = admin_row['full_name'] if admin_row else 'Admin'
                reg_by_role = 'Admin'
                reg_admin_id = session['admin_id']
            elif 'teacher_id' in session:
                teacher_row = conn.execute('SELECT full_name FROM teachers WHERE id = ?', (session['teacher_id'],)).fetchone()
                reg_by = teacher_row['full_name'] if teacher_row else 'Teacher'
                reg_by_role = 'Teacher'

            conn.execute(
                """INSERT OR REPLACE INTO users
                   (username, name, email, user_id, department, phone, role, notes, created_at, registered_by, registered_by_role, registered_by_admin_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (username, username, email, user_id, department, phone, role, notes, datetime.now().isoformat(), reg_by, reg_by_role, reg_admin_id),
            )
            # If class_id provided, add student to that class
            class_id = data.get("class_id", "")
            if class_id:
                try:
                    cid = int(class_id)
                    if cid < 0:
                        # Negative ID = admin_class; use the real admin_classes.id
                        conn.execute(
                            "INSERT OR IGNORE INTO class_students (class_id, student_name) VALUES (?, ?)",
                            (-cid, username),
                        )
                    elif cid > 0 and 'teacher_id' in session:
                        conn.execute(
                            "INSERT OR IGNORE INTO class_students (class_id, student_name) VALUES (?, ?)",
                            (cid, username),
                        )
                except Exception:
                    pass
            conn.commit()
        except Exception as e:
            print(f"Database error: {e}")
        finally:
            conn.close()

        # ── If admin registered a non-Student, create teacher account & send credentials ──
        teacher_created = False
        if 'admin_id' in session and role.lower() != 'student' and user_pin_plain and email:
            if not user_pin_plain.isdigit() or len(user_pin_plain) < 4 or len(user_pin_plain) > 6:
                pass  # skip if invalid PIN
            else:
                pin_hash = hashlib.sha256(user_pin_plain.encode()).hexdigest()
                t_username = email.split("@")[0].lower().replace(" ", "").replace(".", "_")
                conn = get_db_connection()
                existing = conn.execute(
                    "SELECT id FROM teachers WHERE username = ? OR email = ?", (t_username, email)
                ).fetchone()
                if not existing:
                    try:
                        conn.execute(
                            """INSERT INTO teachers (username, password_hash, email, full_name, phone, department, designation, is_verified, user_pin, registered_by_admin_id)
                               VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?)""",
                            (t_username, pin_hash, email, username, phone, department, role, pin_hash, session.get('admin_id', 0)),
                        )
                        new_t = conn.execute("SELECT id FROM teachers WHERE username = ?", (t_username,)).fetchone()
                        teacher_uid = ""
                        if new_t:
                            teacher_uid = generate_user_id("TCH", conn, "teachers")
                            conn.execute("UPDATE teachers SET user_id = ? WHERE id = ?", (teacher_uid, new_t['id']))
                        conn.commit()
                        teacher_created = True
                        # Send credentials email with all details
                        role_label = role if role in ("HOD", "Staff") else "Teacher"
                        # Get admin info for the email
                        admin_row = conn.execute('SELECT full_name, email FROM admins WHERE id = ?', (session.get('admin_id', 0),)).fetchone()
                        a_name = admin_row['full_name'] if admin_row else ''
                        a_email = admin_row['email'] if admin_row else ''
                        cred_body = build_credentials_email(
                            username, teacher_uid, user_pin_plain, role_label,
                            email=email, department=department, designation=role,
                            phone=phone, admin_name=a_name, admin_email=a_email
                        )
                        send_email(email, f"Your {APP_NAME} Account — Login Credentials", cred_body)
                    except sqlite3.IntegrityError:
                        pass
                conn.close()

        return jsonify({
            "success": True,
            "message": f"Successfully registered {username} with {new_count} face samples",
            "totalSamples": len(all_faces),
            "userData": {
                "name": username,
                "email": email,
                "userId": user_id,
                "department": department,
                "role": role,
                "phone": phone,
                "notes": notes,
                "registeredBy": reg_by,
                "registeredByRole": reg_by_role,
            },
        }), 200

    except Exception as e:
        print(f"Upload error: {e}")
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/recognize", methods=["GET"])
@app.route("/admin/gate-scan", methods=["GET"], endpoint="admin_gate_scan")
@app.route("/teacher/class-scan", methods=["GET"], endpoint="teacher_class_scan")
@any_login_required
def recognize():
    classes = []

    # Get classes for the teacher if logged in
    if "teacher_id" in session:
        conn = get_db_connection()
        classes = conn.execute(
            "SELECT * FROM classes WHERE teacher_id = ?", (session["teacher_id"],)
        ).fetchall()
        conn.close()

    return render_template(
        "recognize.html", classes=classes
    )


@app.route("/attendance")
def show_attendance():
    # Get filter parameters from request
    # Convert date format: HTML input uses YYYY-MM-DD, DB uses DD-MM-YYYY
    filter_date_input = request.args.get('date', '')
    
    # If no date provided, default to today in DD-MM-YYYY format
    if not filter_date_input:
        filter_date = datetime.now().strftime("%d-%m-%Y")
        filter_date_html = datetime.now().strftime("%Y-%m-%d")
    else:
        # Convert from YYYY-MM-DD to DD-MM-YYYY for database query
        try:
            date_obj = datetime.strptime(filter_date_input, "%Y-%m-%d")
            filter_date = date_obj.strftime("%d-%m-%Y")
            filter_date_html = filter_date_input
        except:
            filter_date = datetime.now().strftime("%d-%m-%Y")
            filter_date_html = datetime.now().strftime("%Y-%m-%d")
    
    filter_department = request.args.get('department', '')
    filter_name = request.args.get('name', '')
    
    # Initialize empty list for attendance records
    attendance_records = []
    departments = []
    
    try:
        conn = get_db_connection()
        
        # Get all departments for filter dropdown
        all_depts = conn.execute(
            "SELECT DISTINCT department FROM users WHERE department IS NOT NULL AND department != ''"
        ).fetchall()
        departments = [d['department'] for d in all_depts]
        
        # Build query — deduplicate: only the FIRST record per student per day
        # Use MIN(time) + GROUP BY to get single entry per person per day
        query = """
            SELECT 
                ar.student_name,
                ar.date,
                MIN(ar.time) as time,
                ar.status,
                u.user_id,
                u.department,
                u.email
            FROM attendance_records ar
            LEFT JOIN users u ON ar.student_name = u.username
            WHERE ar.student_name != 'Unknown' AND ar.student_name != 'Error'
        """
        
        params = []
        
        # Apply date filter (default to today)
        if filter_date:
            query += " AND ar.date = ?"
            params.append(filter_date)
        
        # Apply department filter
        if filter_department:
            query += " AND u.department = ?"
            params.append(filter_department)
        
        # Apply name filter
        if filter_name:
            query += " AND (ar.student_name LIKE ? OR u.user_id LIKE ?)"
            params.append(f"%{filter_name}%")
            params.append(f"%{filter_name}%")
        
        # GROUP BY to get only one record per student per day
        query += " GROUP BY ar.student_name, ar.date"
        query += " ORDER BY ar.time DESC"
        
        db_records = conn.execute(query, params).fetchall()
        
        # Convert database records to list format
        for record in db_records:
            attendance_records.append({
                'student_name': record['student_name'],
                'date': record['date'],
                'time': record['time'],
                'status': record['status'] or 'Present',
                'user_id': record['user_id'] or 'N/A',
                'department': record['department'] or 'N/A',
                'email': record['email'] or 'N/A'
            })
        
        # Calculate statistics
        today = datetime.now().strftime("%d-%m-%Y")
        total_records = len(attendance_records)
        unique_users = len(set(r['student_name'] for r in attendance_records))
        today_count = sum(1 for r in attendance_records if r['date'] == today)
        
        # Department-wise stats for filtered results
        dept_stats = {}
        for record in attendance_records:
            dept = record['department'] or 'N/A'
            if dept not in dept_stats:
                dept_stats[dept] = 0
            dept_stats[dept] += 1
        
        conn.close()
        
    except Exception as e:
        print(f"Error querying database for attendance records: {str(e)}")
        import traceback
        traceback.print_exc()
        attendance_records = []
        departments = []
        total_records = 0
        unique_users = 0
        today_count = 0
        dept_stats = {}
    
    has_records = len(attendance_records) > 0
    
    stats = {
        'total_records': total_records,
        'unique_users': unique_users,
        'today_count': today_count,
        'dept_stats': dept_stats
    }

    return render_template(
        "attendance.html", 
        attendance_data=attendance_records, 
        has_records=has_records,
        stats=stats,
        departments=departments,
        filter_date=filter_date_html,
        filter_department=filter_department,
        filter_name=filter_name,
        current_date=datetime.now().strftime("%d-%m-%Y")
    )


@app.route("/manage_users")
@app.route("/admin/users", endpoint="admin_users")
@app.route("/teacher/users", endpoint="teacher_users")
@any_login_required
def manage_users():
    # Get all users from names.pkl
    users = get_all_users()
    
    # Determine who is logged in
    is_admin = 'admin_id' in session
    viewer_name = ''
    if is_admin:
        conn_tmp = get_db_connection()
        admin_row = conn_tmp.execute('SELECT full_name FROM admins WHERE id = ?', (session['admin_id'],)).fetchone()
        viewer_name = admin_row['full_name'] if admin_row else ''
        conn_tmp.close()
    elif 'teacher_id' in session:
        conn_tmp = get_db_connection()
        teacher_row = conn_tmp.execute('SELECT full_name FROM teachers WHERE id = ?', (session['teacher_id'],)).fetchone()
        viewer_name = teacher_row['full_name'] if teacher_row else ''
        conn_tmp.close()

    # Get detailed user info from database
    user_details = []
    conn = get_db_connection()
    for username in users:
        user_info = conn.execute(
            "SELECT name, email, user_id, department, role, phone, notes, created_at, registered_by, registered_by_role FROM users WHERE username = ?",
            (username,)
        ).fetchone()
        if user_info:
            # Teacher can only see users they registered
            if not is_admin and user_info['registered_by'] != viewer_name:
                continue
            user_details.append({
                'username': username,
                'name': user_info['name'] if user_info['name'] else username,
                'email': user_info['email'],
                'user_id': user_info['user_id'],
                'department': user_info['department'],
                'role': user_info['role'],
                'phone': user_info['phone'] or '',
                'notes': user_info['notes'] or '',
                'created_at': user_info['created_at'] or '',
                'registered_by': user_info['registered_by'] or '',
                'registered_by_role': user_info['registered_by_role'] or '',
            })
        else:
            # Unlinked users (no DB entry) — only show to admin
            if is_admin:
                user_details.append({
                    'username': username,
                    'name': username,
                    'email': '', 'user_id': '', 'department': '', 'role': '',
                    'phone': '', 'notes': '', 'created_at': '',
                    'registered_by': '', 'registered_by_role': '',
                })
    conn.close()
    
    return render_template("manage_users.html", users=users, user_details=user_details, is_admin=is_admin)


@app.route("/delete_user/<username>", methods=["POST"])
def delete_user_route(username):
    target = 'admin_users' if 'admin_id' in session else 'teacher_users' if 'teacher_id' in session else 'manage_users'
    if delete_user(username):
        flash(f"User '{username}' has been deleted successfully.", "success")
        return redirect(url_for(target))
    flash(f"Error deleting user '{username}'.", "danger")
    return redirect(url_for(target))


@app.route("/api/edit_user/<username>", methods=["POST"])
def edit_user_route(username):
    """Edit user details in the database."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        name = data.get("name", "").strip()
        email = data.get("email", "").strip()
        user_id = data.get("user_id", "").strip()
        department = data.get("department", "").strip()
        role = data.get("role", "").strip()

        if not name:
            return jsonify({"error": "Name is required"}), 400

        conn = get_db_connection()
        # Check if user exists
        existing = conn.execute(
            "SELECT id FROM users WHERE username = ?", (username,)
        ).fetchone()

        if not existing:
            conn.close()
            return jsonify({"error": "User not found"}), 404

        conn.execute(
            """UPDATE users SET name = ?, email = ?, user_id = ?, department = ?, role = ?
               WHERE username = ?""",
            (name, email, user_id, department, role, username),
        )
        conn.commit()
        conn.close()

        return jsonify({
            "success": True,
            "message": f"User '{name}' updated successfully",
            "user": {
                "username": username,
                "name": name,
                "email": email,
                "user_id": user_id,
                "department": department,
                "role": role,
            },
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/settings", methods=["GET", "POST"])
@app.route("/admin/settings", methods=["GET", "POST"], endpoint="admin_settings")
@admin_login_required
def settings():
    if request.method == "POST":
        # Check if this is a JSON request (AJAX from settings page)
        if request.is_json:
            data = request.get_json()
            action = data.get('action', '')
            
            current = load_settings()
            
            if action == 'add_department':
                dept = (data.get('department') or '').strip()
                if dept and dept not in current.get('departments', []):
                    current.setdefault('departments', []).append(dept)
                    save_settings(current)
                    return jsonify({'success': True, 'departments': current['departments']})
                return jsonify({'success': False, 'message': 'Department empty or already exists'})
            
            elif action == 'remove_department':
                dept = data.get('department', '')
                depts = current.get('departments', [])
                if dept in depts:
                    depts.remove(dept)
                    current['departments'] = depts
                    save_settings(current)
                    return jsonify({'success': True, 'departments': current['departments']})
                return jsonify({'success': False, 'message': 'Department not found'})
            
            elif action == 'add_role':
                role = (data.get('role') or '').strip()
                if role and role not in current.get('roles', []):
                    current.setdefault('roles', []).append(role)
                    save_settings(current)
                    return jsonify({'success': True, 'roles': current['roles']})
                return jsonify({'success': False, 'message': 'Role empty or already exists'})
            
            elif action == 'remove_role':
                role = data.get('role', '')
                roles = current.get('roles', [])
                if role in roles:
                    roles.remove(role)
                    current['roles'] = roles
                    save_settings(current)
                    return jsonify({'success': True, 'roles': current['roles']})
                return jsonify({'success': False, 'message': 'Role not found'})
            
            elif action == 'clear_attendance':
                try:
                    conn = get_db_connection()
                    conn.execute("DELETE FROM attendance_records")
                    conn.commit()
                    conn.close()
                    return jsonify({'success': True, 'message': 'All attendance records cleared'})
                except Exception as e:
                    return jsonify({'success': False, 'message': str(e)})
            
            elif action == 'reset_settings':
                if os.path.exists("data/settings.pkl"):
                    os.remove("data/settings.pkl")
                return jsonify({'success': True, 'message': 'Settings reset to defaults'})
            
            return jsonify({'success': False, 'message': 'Unknown action'})
        
        # Form-based settings save
        current = load_settings()
        new_settings = {
            "camera_index": int(request.form.get("camera_index", 0)),
            "required_samples": int(request.form.get("required_samples", 50)),
            "attendance_threshold": float(request.form.get("attendance_threshold", 0.6)),
            "min_confidence": int(request.form.get("min_confidence", 50)),
            "auto_save_attendance": request.form.get("auto_save_attendance") == "on",
            "recognition_interval": int(request.form.get("recognition_interval", 2000)),
            "duplicate_check": request.form.get("duplicate_check") == "on",
            "notification_sound": request.form.get("notification_sound") == "on",
            "departments": current.get('departments', []),
            "roles": current.get('roles', []),
        }
        save_settings(new_settings)
        flash("Settings saved successfully!", "success")
        return redirect(url_for("admin_settings"))

    current_settings = load_settings()
    
    # Get system stats for the settings page
    conn = get_db_connection()
    total_records = conn.execute("SELECT COUNT(*) as c FROM attendance_records").fetchone()['c']
    total_users = len(get_all_users())
    conn.close()
    
    sys_stats = {
        'total_records': total_records,
        'total_users': total_users,
        'model_exists': os.path.exists("data/face_recognizer.pkl"),
        'faces_data_exists': os.path.exists("data/faces_data.pkl"),
    }
    
    return render_template("settings.html", settings=current_settings, sys_stats=sys_stats)


# ── Admin Class Management API ──
@app.route("/api/admin/classes", methods=["GET"])
@admin_login_required
def api_admin_classes():
    """Get all admin-managed classes with assignment info."""
    conn = get_db_connection()
    classes = conn.execute(
        "SELECT * FROM admin_classes ORDER BY institution_type, name"
    ).fetchall()
    result = []
    for c in classes:
        teachers = conn.execute(
            """SELECT t.id, t.full_name, t.user_id, t.department, t.designation
               FROM teacher_class_assignments tca
               JOIN teachers t ON t.id = tca.teacher_id
               WHERE tca.admin_class_id = ?""",
            (c["id"],),
        ).fetchall()
        result.append({
            "id": c["id"],
            "name": c["name"],
            "institution_type": c["institution_type"],
            "department": c["department"] or "",
            "description": c["description"] or "",
            "teachers": [{"id": t["id"], "name": t["full_name"], "user_id": t["user_id"] or "",
                          "department": t["department"] or "", "designation": t["designation"] or ""} for t in teachers],
        })
    conn.close()
    return jsonify({"success": True, "classes": result})


@app.route("/api/admin/classes", methods=["POST"])
@admin_login_required
def api_admin_classes_create():
    """Create a new admin-managed class."""
    data = request.get_json()
    name = (data.get("name") or "").strip()
    inst_type = (data.get("institution_type") or "college").strip().lower()
    department = (data.get("department") or "").strip()
    description = (data.get("description") or "").strip()

    if not name:
        return jsonify({"success": False, "message": "Class name is required"}), 400
    if inst_type not in ("school", "college"):
        inst_type = "college"

    conn = get_db_connection()
    conn.execute(
        "INSERT INTO admin_classes (name, institution_type, department, description) VALUES (?, ?, ?, ?)",
        (name, inst_type, department, description),
    )
    conn.commit()
    conn.close()
    return jsonify({"success": True, "message": f"Class '{name}' created"})


@app.route("/api/admin/classes/<int:class_id>", methods=["DELETE"])
@admin_login_required
def api_admin_classes_delete(class_id):
    """Delete an admin-managed class."""
    conn = get_db_connection()
    conn.execute("DELETE FROM teacher_class_assignments WHERE admin_class_id = ?", (class_id,))
    conn.execute("DELETE FROM admin_classes WHERE id = ?", (class_id,))
    conn.commit()
    conn.close()
    return jsonify({"success": True, "message": "Class deleted"})


@app.route("/api/admin/classes/<int:class_id>/assign", methods=["POST"])
@admin_login_required
def api_admin_class_assign(class_id):
    """Assign a teacher to an admin-managed class."""
    data = request.get_json()
    teacher_id = data.get("teacher_id")
    if not teacher_id:
        return jsonify({"success": False, "message": "Teacher ID is required"}), 400

    conn = get_db_connection()
    # Verify class exists
    cls = conn.execute("SELECT id, institution_type FROM admin_classes WHERE id = ?", (class_id,)).fetchone()
    if not cls:
        conn.close()
        return jsonify({"success": False, "message": "Class not found"}), 404

    # Verify teacher exists and get designation
    teacher = conn.execute("SELECT id, designation FROM teachers WHERE id = ?", (teacher_id,)).fetchone()
    if not teacher:
        conn.close()
        return jsonify({"success": False, "message": "Teacher not found"}), 404

    # For college: allow HOD and Teacher. For school: allow Teacher only.
    designation = (teacher["designation"] or "").strip().lower()
    if cls["institution_type"] == "school" and designation == "hod":
        conn.close()
        return jsonify({"success": False, "message": "School classes can only be assigned to Teachers"}), 400

    try:
        conn.execute(
            "INSERT INTO teacher_class_assignments (admin_class_id, teacher_id) VALUES (?, ?)",
            (class_id, int(teacher_id)),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        return jsonify({"success": False, "message": "Teacher already assigned to this class"}), 409
    conn.close()
    return jsonify({"success": True, "message": "Teacher assigned successfully"})


@app.route("/api/admin/classes/<int:class_id>/unassign/<int:teacher_id>", methods=["DELETE"])
@admin_login_required
def api_admin_class_unassign(class_id, teacher_id):
    """Remove a teacher from an admin-managed class."""
    conn = get_db_connection()
    conn.execute(
        "DELETE FROM teacher_class_assignments WHERE admin_class_id = ? AND teacher_id = ?",
        (class_id, teacher_id),
    )
    conn.commit()
    conn.close()
    return jsonify({"success": True, "message": "Teacher unassigned"})


@app.route("/api/admin/teachers", methods=["GET"])
@admin_login_required
def api_admin_teachers_list():
    """Get all teachers for assignment dropdown."""
    conn = get_db_connection()
    teachers = conn.execute(
        "SELECT id, full_name, user_id, department, designation FROM teachers ORDER BY full_name"
    ).fetchall()
    conn.close()
    return jsonify({
        "success": True,
        "teachers": [{"id": t["id"], "name": t["full_name"], "user_id": t["user_id"] or "",
                       "department": t["department"] or "", "designation": t["designation"] or ""} for t in teachers],
    })


@app.route("/api/classes/for_capture", methods=["GET"])
@any_login_required
def api_classes_for_capture():
    """Get classes available for the current user on the capture page."""
    conn = get_db_connection()
    classes = []
    if "teacher_id" in session:
        # Teacher's own classes
        own = conn.execute(
            "SELECT id, name, 'own' as source FROM classes WHERE teacher_id = ?",
            (session["teacher_id"],),
        ).fetchall()
        # Admin-assigned classes
        assigned = conn.execute(
            """SELECT ac.id, ac.name, 'assigned' as source
               FROM admin_classes ac
               JOIN teacher_class_assignments tca ON tca.admin_class_id = ac.id
               WHERE tca.teacher_id = ?""",
            (session["teacher_id"],),
        ).fetchall()
        classes = [{"id": r["id"], "name": r["name"], "source": r["source"]} for r in own]
        classes += [{"id": -r["id"], "name": r["name"] + " ★", "source": r["source"]} for r in assigned]
    elif "admin_id" in session:
        # All admin-managed classes
        rows = conn.execute("SELECT id, name FROM admin_classes ORDER BY name").fetchall()
        classes = [{"id": -r["id"], "name": r["name"], "source": "admin"} for r in rows]
    conn.close()
    return jsonify({"success": True, "classes": classes})


# API Routes for Real-time Face Recognition
@app.route("/api/capture_frame", methods=["POST"])
def capture_frame():
    """Receive and process a captured frame from the client"""
    try:
        data = request.get_json()
        frame_data = data.get('frame')
        
        if not frame_data:
            return jsonify({'success': False, 'message': 'No frame data received'})
        
        # Decode the base64 image
        frame_bytes = base64.b64decode(frame_data.split(',')[1])
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'success': False, 'message': 'Failed to decode frame'})
        
        # Detect faces in the frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        faces = facedetect.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(20, 20)
        )
        
        faces_detected = len(faces)
        recognition_results = []
        
        if faces_detected > 0 and os.path.exists("data/names.pkl") and os.path.exists("data/faces_data.pkl"):
            try:
                # Load face data
                with open("data/names.pkl", "rb") as f:
                    names = pickle.load(f)
                with open("data/faces_data.pkl", "rb") as f:
                    faces_data = pickle.load(f)
                
                faces_data = faces_data.reshape(faces_data.shape[0], -1)
                
                # Train classifier
                n_neighbors = min(3, faces_data.shape[0])
                knn = KNeighborsClassifier(n_neighbors=n_neighbors)
                knn.fit(faces_data, names)
                
                # Process each detected face
                for (x, y, w, h) in faces:
                    crop_img = frame[y:y+h, x:x+w]
                    crop_img_resized = cv2.resize(crop_img, (50, 50))
                    crop_img_resized_flat = crop_img_resized.reshape(1, -1)
                    
                    # Get prediction
                    output = knn.predict(crop_img_resized_flat)
                    confidence = knn.predict_proba(crop_img_resized_flat)
                    confidence_score = max(confidence[0]) * 100
                    
                    recognition_results.append({
                        'name': str(output[0]),
                        'confidence': round(confidence_score, 2),
                        'x': int(x),
                        'y': int(y),
                        'w': int(w),
                        'h': int(h)
                    })
            except Exception as e:
                print(f"Error in recognition: {str(e)}")
        
        return jsonify({
            'success': True,
            'faces_detected': faces_detected,
            'results': recognition_results
        })
    
    except Exception as e:
        print(f"Error in capture_frame: {str(e)}")
        return jsonify({'success': False, 'message': str(e)})


@app.route("/api/save_attendance", methods=["POST"])
def save_attendance_api():
    """Save recognized attendance — supports both gate (admin) and class (teacher) modes.
    A student can be marked present TWICE per day: once at gate, once in class."""
    try:
        data = request.get_json()
        name = (data.get('name') or '').strip()
        class_id = data.get('class_id') or None
        attendance_type = data.get('attendance_type', 'gate')  # 'gate' or 'class'

        # Normalise empty string class_id to None
        if class_id == '':
            class_id = None

        if not name or name in ('Unknown', 'Error'):
            return jsonify({'success': False, 'message': 'No valid name provided'})

        # Determine who is recording
        teacher_id = None
        admin_id = None
        if attendance_type == 'class' and "teacher_id" in session:
            teacher_id = session["teacher_id"]
        elif attendance_type == 'gate' and "admin_id" in session:
            admin_id = session["admin_id"]
        else:
            # Fallback: use provided teacher_id or default
            teacher_id = data.get('teacher_id', 1)

        conn = get_db_connection()
        now = datetime.now()
        date_str = now.strftime("%d-%m-%Y")
        time_str = now.strftime("%H:%M:%S")

        # Check duplicate: same student + same date + same attendance_type
        # (allows 1 gate + 1 class entry per day)
        try:
            existing_record = conn.execute(
                "SELECT * FROM attendance_records WHERE student_name = ? AND date = ? AND attendance_type = ?",
                (name, date_str, attendance_type)
            ).fetchone()

            if existing_record:
                conn.close()
                type_label = "Gate Entry" if attendance_type == "gate" else "Class Entry"
                return jsonify({
                    'success': False,
                    'message': f'✓ {name} already has {type_label} for today at {existing_record["time"]}',
                    'duplicate': True,
                    'already_recorded': True
                })
        except Exception as e:
            print(f"Error checking existing records: {str(e)}")

        # Record the attendance
        try:
            conn.execute(
                "INSERT INTO attendance_records (student_name, date, time, status, class_id, teacher_id, admin_id, attendance_type) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (name, date_str, time_str, "Present", class_id, teacher_id, admin_id, attendance_type)
            )
            conn.commit()
            type_label = "Gate" if attendance_type == "gate" else "Class"
            print(f"✓ {type_label} attendance recorded for {name} at {time_str}")
        except sqlite3.IntegrityError:
            conn.close()
            return jsonify({
                'success': False,
                'message': f'Attendance already recorded for {name} today',
                'duplicate': True,
                'already_recorded': True
            })

        # Save to CSV
        attendance_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Attendance")
        os.makedirs(attendance_dir, exist_ok=True)

        csv_file = os.path.join(attendance_dir, f"Attendance_{date_str}.csv")
        file_exists = os.path.isfile(csv_file)
        try:
            with open(csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["NAME", "DATE", "TIME", "TYPE"])
                writer.writerow([name, date_str, time_str, attendance_type])
        except Exception as csv_error:
            print(f"Error saving to CSV: {str(csv_error)}")

        # Fetch user details for response
        user_info = conn.execute(
            "SELECT user_id, department, email, role FROM users WHERE username = ?",
            (name,)
        ).fetchone()
        conn.close()

        type_label = "Gate Entry" if attendance_type == "gate" else "Class Entry"
        return jsonify({
            'success': True,
            'message': f'✓ {type_label} recorded for {name}',
            'attendance_type': attendance_type,
            'data': {
                'name': name,
                'date': date_str,
                'time': time_str,
                'attendance_type': attendance_type,
                'user_id': user_info['user_id'] if user_info else 'N/A',
                'department': user_info['department'] if user_info else 'N/A',
                'email': user_info['email'] if user_info else 'N/A',
                'role': user_info['role'] if user_info else 'N/A'
            }
        })

    except Exception as e:
        print(f"Error in save_attendance_api: {str(e)}")
        return jsonify({'success': False, 'message': str(e)})


@app.route("/api/get_recognized_users", methods=["GET"])
def get_recognized_users():
    """Get list of all registered users for dropdown"""
    try:
        if not os.path.exists("data/names.pkl"):
            return jsonify({'success': False, 'users': []})
        
        with open("data/names.pkl", "rb") as f:
            names = pickle.load(f)
        
        return jsonify({
            'success': True,
            'users': list(set(names))  # Remove duplicates
        })
    
    except Exception as e:
        print(f"Error in get_recognized_users: {str(e)}")
        return jsonify({'success': False, 'users': []})


@app.route("/export_attendance", methods=["GET", "POST"])
@app.route("/admin/export", methods=["GET", "POST"], endpoint="admin_export")
@app.route("/teacher/export-data", methods=["GET", "POST"], endpoint="teacher_export")
@any_login_required
def export_attendance():
    if request.method == "POST":
        import io, json as jsonlib
        
        format_type = request.form.get('format', 'csv')
        start_date = request.form.get('start_date', '')
        end_date = request.form.get('end_date', '')
        department_filter = request.form.get('department', '')
        name_filter = request.form.get('name', '')
        include_headers = request.form.get('include_headers') == 'on'
        include_stats = request.form.get('include_stats') == 'on'
        deduplicate = request.form.get('deduplicate') == 'on'
        
        conn = get_db_connection()
        
        # Scope filter for teacher vs admin
        scope_clause = ""
        scope_params = []
        if "teacher_id" in session and "admin_id" not in session:
            scope_clause = " AND ar.teacher_id = ?"
            scope_params = [session["teacher_id"]]
        
        # Build query with proper user details
        if deduplicate:
            query = """
                SELECT ar.student_name, ar.date, MIN(ar.time) as time, ar.status,
                       u.user_id, u.department, u.email, u.role, u.phone
                FROM attendance_records ar
                LEFT JOIN users u ON ar.student_name = u.username
                WHERE ar.student_name != 'Unknown' AND ar.student_name != 'Error'
            """ + scope_clause
        else:
            query = """
                SELECT ar.student_name, ar.date, ar.time, ar.status,
                       u.user_id, u.department, u.email, u.role, u.phone
                FROM attendance_records ar
                LEFT JOIN users u ON ar.student_name = u.username
                WHERE ar.student_name != 'Unknown' AND ar.student_name != 'Error'
            """ + scope_clause
        params = list(scope_params)
        
        # Date range filtering — convert YYYY-MM-DD inputs to DD-MM-YYYY for DB
        if start_date:
            try:
                start_obj = datetime.strptime(start_date, "%Y-%m-%d")
                # Get all records and filter in Python for correct date comparison
            except:
                pass
        if end_date:
            try:
                end_obj = datetime.strptime(end_date, "%Y-%m-%d")
            except:
                pass
        
        if department_filter:
            query += " AND u.department = ?"
            params.append(department_filter)
        
        if name_filter:
            query += " AND (ar.student_name LIKE ? OR u.user_id LIKE ?)"
            params.append(f"%{name_filter}%")
            params.append(f"%{name_filter}%")
        
        if deduplicate:
            query += " GROUP BY ar.student_name, ar.date"
        
        query += " ORDER BY ar.date DESC, ar.time DESC"
        
        all_records = conn.execute(query, params).fetchall()
        
        # Filter by date range in Python (since DD-MM-YYYY string comparison doesn't work)
        records = []
        for r in all_records:
            try:
                rec_date = datetime.strptime(r['date'], "%d-%m-%Y")
                if start_date:
                    start_obj = datetime.strptime(start_date, "%Y-%m-%d")
                    if rec_date < start_obj:
                        continue
                if end_date:
                    end_obj = datetime.strptime(end_date, "%Y-%m-%d")
                    if rec_date > end_obj:
                        continue
            except:
                pass
            records.append(r)
        
        conn.close()
        
        if format_type == 'csv':
            output = io.StringIO()
            writer = csv.writer(output)
            
            if include_headers:
                writer.writerow(["Name", "User ID", "Department", "Email", "Role", "Date", "Time", "Status"])
            
            for record in records:
                writer.writerow([
                    record['student_name'],
                    record['user_id'] or '',
                    record['department'] or '',
                    record['email'] or '',
                    record['role'] or '',
                    record['date'],
                    record['time'],
                    record['status'] or 'Present'
                ])
            
            if include_stats:
                writer.writerow([])
                writer.writerow(["--- Statistics ---"])
                writer.writerow(["Total Records", len(records)])
                unique = len(set(r['student_name'] for r in records))
                writer.writerow(["Unique Users", unique])
                writer.writerow(["Export Date", datetime.now().strftime("%d-%m-%Y %H:%M:%S")])
                # Dept breakdown
                dept_counts = {}
                for r in records:
                    d = r['department'] or 'Unknown'
                    dept_counts[d] = dept_counts.get(d, 0) + 1
                writer.writerow([])
                writer.writerow(["Department", "Count"])
                for dept, count in sorted(dept_counts.items()):
                    writer.writerow([dept, count])
            
            response = make_response(output.getvalue())
            response.headers["Content-Disposition"] = f"attachment;filename=attendance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            response.headers["Content-Type"] = "text/csv"
            return response
        
        elif format_type == 'json':
            data = {
                'records': [{
                    'name': r['student_name'],
                    'user_id': r['user_id'] or '',
                    'department': r['department'] or '',
                    'email': r['email'] or '',
                    'role': r['role'] or '',
                    'date': r['date'],
                    'time': r['time'],
                    'status': r['status'] or 'Present'
                } for r in records]
            }
            if include_stats:
                unique = len(set(r['student_name'] for r in records))
                dept_counts = {}
                for r in records:
                    d = r['department'] or 'Unknown'
                    dept_counts[d] = dept_counts.get(d, 0) + 1
                data['statistics'] = {
                    'total_records': len(records),
                    'unique_users': unique,
                    'department_breakdown': dept_counts,
                    'export_date': datetime.now().isoformat()
                }
            
            response = make_response(jsonlib.dumps(data, indent=2))
            response.headers["Content-Disposition"] = f"attachment;filename=attendance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            response.headers["Content-Type"] = "application/json"
            return response
        
        else:
            flash("Only CSV and JSON formats are supported.", "info")
            export_target = 'admin_export' if 'admin_id' in session else 'teacher_export' if 'teacher_id' in session else 'export_attendance'
            return redirect(url_for(export_target))
    
    # GET — show the export form with stats and filter options
    conn = get_db_connection()
    
    # Scope queries for teacher vs admin
    scope_where = ""
    scope_params_get = []
    if "teacher_id" in session and "admin_id" not in session:
        scope_where = " AND ar.teacher_id = ?"
        scope_params_get = [session["teacher_id"]]
    
    total_records = conn.execute(
        "SELECT COUNT(*) as count FROM attendance_records ar WHERE ar.student_name != 'Unknown' AND ar.student_name != 'Error'" + scope_where,
        scope_params_get
    ).fetchone()['count']
    
    # Get departments for filter
    all_depts = conn.execute(
        "SELECT DISTINCT department FROM users WHERE department IS NOT NULL AND department != ''"
    ).fetchall()
    departments = [d['department'] for d in all_depts]
    
    # Get date range
    date_range = conn.execute(
        "SELECT MIN(ar.date) as earliest, MAX(ar.date) as latest FROM attendance_records ar WHERE 1=1" + scope_where,
        scope_params_get
    ).fetchone()
    
    # Get unique students count
    unique_count = conn.execute(
        "SELECT COUNT(DISTINCT ar.student_name) as count FROM attendance_records ar WHERE ar.student_name != 'Unknown' AND ar.student_name != 'Error'" + scope_where,
        scope_params_get
    ).fetchone()['count']
    
    # Get today's count
    today = datetime.now().strftime("%d-%m-%Y")
    today_count = conn.execute(
        "SELECT COUNT(DISTINCT ar.student_name) as count FROM attendance_records ar WHERE ar.date = ? AND ar.student_name != 'Unknown'" + scope_where,
        [today] + scope_params_get
    ).fetchone()['count']
    
    conn.close()
    
    # Total users count (scoped for teacher)
    if "teacher_id" in session and "admin_id" not in session:
        conn_u = get_db_connection()
        teacher_row = conn_u.execute('SELECT full_name FROM teachers WHERE id = ?', (session['teacher_id'],)).fetchone()
        teacher_name = teacher_row['full_name'] if teacher_row else ''
        total_users_count = conn_u.execute(
            "SELECT COUNT(*) as count FROM users WHERE registered_by = ?", (teacher_name,)
        ).fetchone()['count']
        conn_u.close()
    else:
        total_users_count = len(get_all_users())
    
    stats = {
        'total_records': total_records,
        'total_users': total_users_count,
        'unique_recorded': unique_count,
        'today_count': today_count,
        'earliest_date': date_range['earliest'] if date_range else None,
        'latest_date': date_range['latest'] if date_range else None,
    }
    
    # Get classes for teacher portal
    classes = []
    if "teacher_id" in session and "admin_id" not in session:
        conn2 = get_db_connection()
        classes = conn2.execute(
            "SELECT id, name FROM classes WHERE teacher_id = ?", (session["teacher_id"],)
        ).fetchall()
        conn2.close()
    
    return render_template("export_attendance.html", stats=stats, departments=departments, classes=classes)


# API endpoint to get departments list (used by multiple pages)
@app.route("/api/departments")
def get_departments():
    conn = get_db_connection()
    depts = conn.execute(
        "SELECT DISTINCT department FROM users WHERE department IS NOT NULL AND department != '' ORDER BY department"
    ).fetchall()
    conn.close()
    return jsonify({'departments': [d['department'] for d in depts]})


# Auto-generate next User/Staff ID based on role
@app.route("/api/next_user_id")
@any_login_required
def api_next_user_id():
    """Generate next user ID based on role: STU-XXXX, STF-XXXX, TCH-XXXX, etc."""
    role = request.args.get("role", "Student").strip()
    prefix_map = {"Student": "STU", "Teacher": "TCH", "Staff": "STF", "Admin": "ADM"}
    prefix = prefix_map.get(role, "USR")

    conn = get_db_connection()
    result = conn.execute(
        "SELECT user_id FROM users WHERE user_id LIKE ? ORDER BY id DESC LIMIT 1",
        (f"{prefix}-%",)
    ).fetchone()
    if result and result['user_id']:
        try:
            last_num = int(result['user_id'].split('-')[1])
            next_id = f"{prefix}-{last_num + 1:04d}"
        except (ValueError, IndexError):
            count = conn.execute("SELECT COUNT(*) as c FROM users WHERE user_id LIKE ?", (f"{prefix}-%",)).fetchone()['c']
            next_id = f"{prefix}-{count + 1:04d}"
    else:
        next_id = f"{prefix}-0001"
    conn.close()
    return jsonify({"user_id": next_id})


# ── Manage Users API (for settings page) ──
@app.route("/api/admin/all_users", methods=["GET"])
@admin_login_required
def api_admin_all_users():
    """Get all registered users (from users table) for management UI."""
    conn = get_db_connection()
    rows = conn.execute(
        "SELECT id, username, name, email, user_id, department, phone, role, notes, created_at, registered_by, registered_by_role, is_active FROM users ORDER BY id DESC"
    ).fetchall()
    conn.close()
    users = []
    for r in rows:
        is_active = r["is_active"] if "is_active" in r.keys() else 1
        users.append({
            "id": r["id"], "username": r["username"], "name": r["name"],
            "email": r["email"], "user_id": r["user_id"], "department": r["department"],
            "phone": r["phone"] or "", "role": r["role"], "notes": r["notes"] or "",
            "created_at": r["created_at"] or "", "registered_by": r["registered_by"] or "",
            "registered_by_role": r["registered_by_role"] or "",
            "is_active": is_active,
        })
    return jsonify({"success": True, "users": users})


@app.route("/api/admin/users/<int:user_id>/update", methods=["POST"])
@admin_login_required
def api_admin_update_user(user_id):
    """Update a registered user's details."""
    data = request.get_json()
    name = (data.get("name") or "").strip()
    email = (data.get("email") or "").strip()
    department = (data.get("department") or "").strip()
    phone = (data.get("phone") or "").strip()
    role = (data.get("role") or "").strip()
    notes = (data.get("notes") or "").strip()
    new_user_id = (data.get("user_id") or "").strip()

    if not name or not email:
        return jsonify({"success": False, "message": "Name and email are required"}), 400

    conn = get_db_connection()
    existing = conn.execute("SELECT id FROM users WHERE id = ?", (user_id,)).fetchone()
    if not existing:
        conn.close()
        return jsonify({"success": False, "message": "User not found"}), 404

    # Check user_id uniqueness (exclude self)
    if new_user_id:
        dup = conn.execute("SELECT id FROM users WHERE user_id = ? AND id != ?", (new_user_id, user_id)).fetchone()
        if dup:
            conn.close()
            return jsonify({"success": False, "message": "User ID already in use"}), 409

    conn.execute(
        """UPDATE users SET name = ?, email = ?, department = ?, phone = ?,
           role = ?, notes = ?, user_id = ? WHERE id = ?""",
        (name, email, department, phone, role, notes, new_user_id, user_id),
    )
    conn.commit()
    conn.close()
    return jsonify({"success": True, "message": "User updated successfully"})


@app.route("/api/admin/users/<int:user_id>/toggle_active", methods=["POST"])
@admin_login_required
def api_admin_toggle_active(user_id):
    """Toggle active/inactive status for a user."""
    conn = get_db_connection()
    user = conn.execute("SELECT id, is_active FROM users WHERE id = ?", (user_id,)).fetchone()
    if not user:
        conn.close()
        return jsonify({"success": False, "message": "User not found"}), 404
    new_status = 0 if user["is_active"] else 1
    conn.execute("UPDATE users SET is_active = ? WHERE id = ?", (new_status, user_id))
    conn.commit()
    conn.close()
    return jsonify({"success": True, "is_active": new_status, "message": "User " + ("activated" if new_status else "deactivated")})


@app.route("/api/admin/users/<int:user_id>/delete", methods=["DELETE"])
@admin_login_required
def api_admin_delete_user(user_id):
    """Delete a registered user from the database and remove their face data."""
    conn = get_db_connection()
    user = conn.execute("SELECT id, username, name FROM users WHERE id = ?", (user_id,)).fetchone()
    if not user:
        conn.close()
        return jsonify({"success": False, "message": "User not found"}), 404

    username = user["username"]
    # Remove from users table
    conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
    # Remove from class_students
    conn.execute("DELETE FROM class_students WHERE student_name = ?", (username,))
    conn.commit()
    conn.close()

    # Remove from face data (names.pkl and faces_data.pkl)
    try:
        if os.path.exists("data/names.pkl") and os.path.exists("data/faces_data.pkl"):
            with open("data/names.pkl", "rb") as f:
                names = pickle.load(f)
            with open("data/faces_data.pkl", "rb") as f:
                faces = pickle.load(f)
            # Filter out this user's data
            indices = [i for i, n in enumerate(names) if n != username]
            if indices:
                new_names = [names[i] for i in indices]
                new_faces = faces[indices]
                with open("data/names.pkl", "wb") as f:
                    pickle.dump(new_names, f)
                with open("data/faces_data.pkl", "wb") as f:
                    pickle.dump(new_faces, f)
                # Retrain model
                n_neighbors = min(5, len(new_faces))
                knn = KNeighborsClassifier(n_neighbors=n_neighbors)
                knn.fit(new_faces.reshape(new_faces.shape[0], -1), new_names)
                with open("data/face_recognizer.pkl", "wb") as f:
                    pickle.dump(knn, f)
            else:
                # No users left, remove files
                for fp in ["data/names.pkl", "data/faces_data.pkl", "data/face_recognizer.pkl"]:
                    if os.path.exists(fp):
                        os.remove(fp)
    except Exception as e:
        print(f"Error removing face data for {username}: {e}")

    return jsonify({"success": True, "message": f"User '{username}' deleted successfully"})


# Combined API for dropdown options — merges settings + database
@app.route("/api/options")
def get_options():
    """Returns merged department and role lists from settings + database"""
    s = load_settings()
    setting_depts = s.get('departments', [])
    setting_roles = s.get('roles', [])
    
    # Also get departments actually used in DB
    conn = get_db_connection()
    db_depts = conn.execute(
        "SELECT DISTINCT department FROM users WHERE department IS NOT NULL AND department != '' ORDER BY department"
    ).fetchall()
    conn.close()
    
    # Merge (settings list + any DB-only depts)
    all_depts = list(setting_depts)
    for d in db_depts:
        if d['department'] not in all_depts:
            all_depts.append(d['department'])
    
    return jsonify({
        'departments': all_depts,
        'roles': setting_roles,
        'settings': {
            'min_confidence': s.get('min_confidence', 50),
            'recognition_interval': s.get('recognition_interval', 2000),
            'auto_save_attendance': s.get('auto_save_attendance', True),
            'required_samples': s.get('required_samples', 50),
        }
    })


# Export preview API
@app.route("/api/export_preview")
def export_preview():
    """Return a preview of records matching the export filters (max 20 rows)"""
    start_date = request.args.get('start_date', '')
    end_date = request.args.get('end_date', '')
    department_filter = request.args.get('department', '')
    name_filter = request.args.get('name', '')
    
    conn = get_db_connection()
    query = """
        SELECT ar.student_name, ar.date, MIN(ar.time) as time, ar.status,
               u.user_id, u.department, u.email
        FROM attendance_records ar
        LEFT JOIN users u ON ar.student_name = u.username
        WHERE ar.student_name != 'Unknown' AND ar.student_name != 'Error'
    """
    params = []
    
    if department_filter:
        query += " AND u.department = ?"
        params.append(department_filter)
    if name_filter:
        query += " AND (ar.student_name LIKE ? OR u.user_id LIKE ?)"
        params.append(f"%{name_filter}%")
        params.append(f"%{name_filter}%")
    
    query += " GROUP BY ar.student_name, ar.date ORDER BY ar.date DESC, ar.time DESC"
    
    all_records = conn.execute(query, params).fetchall()
    conn.close()
    
    # Filter by date range in Python
    filtered = []
    for r in all_records:
        try:
            rec_date = datetime.strptime(r['date'], "%d-%m-%Y")
            if start_date:
                if rec_date < datetime.strptime(start_date, "%Y-%m-%d"):
                    continue
            if end_date:
                if rec_date > datetime.strptime(end_date, "%Y-%m-%d"):
                    continue
        except:
            pass
        filtered.append({
            'name': r['student_name'],
            'user_id': r['user_id'] or '',
            'department': r['department'] or '',
            'email': r['email'] or '',
            'date': r['date'],
            'time': r['time'],
        })
    
    return jsonify({
        'total': len(filtered),
        'records': filtered[:20]  # Preview limited to 20
    })


# Teacher authentication routes
@app.route("/auth/register", methods=["GET", "POST"])
def teacher_register():
    # Teacher self-signup is disabled — only admin can create teacher accounts
    flash("Teacher accounts are created by the admin. Please contact your administrator.", "info")
    return redirect(url_for("teacher_login"))


@app.route("/auth/verify", methods=["GET", "POST"])
def verify_email():
    email = session.get("pending_verify_email")
    if not email:
        flash("No pending verification. Please register first.", "warning")
        return redirect(url_for("teacher_register"))

    if request.method == "POST":
        otp_input = request.form.get("otp", "").strip()

        if not otp_input:
            flash("Please enter the verification code.", "danger")
            return render_template("auth/verify.html", email=email)

        conn = get_db_connection()
        record = conn.execute(
            """SELECT * FROM email_verifications
               WHERE email = ? AND otp_code = ? AND verified = 0
               ORDER BY created_at DESC LIMIT 1""",
            (email, otp_input),
        ).fetchone()

        if not record:
            conn.close()
            flash("Invalid verification code.", "danger")
            return render_template("auth/verify.html", email=email)

        # Check expiry
        expires = datetime.strptime(record["expires_at"], "%Y-%m-%d %H:%M:%S")
        if datetime.now() > expires:
            conn.close()
            flash("Verification code has expired. Please register again.", "danger")
            return redirect(url_for("teacher_register"))

        # OTP valid — create the teacher account
        try:
            conn.execute(
                """INSERT INTO teachers (username, password_hash, email, full_name, is_verified)
                   VALUES (?, ?, ?, ?, 1)""",
                (
                    record["username"],
                    record["password_hash"],
                    record["email"],
                    record["full_name"],
                ),
            )
            # Auto-generate user_id for the new teacher
            new_teacher = conn.execute(
                "SELECT id FROM teachers WHERE username = ?", (record["username"],)
            ).fetchone()
            if new_teacher:
                teacher_user_id = generate_user_id("TCH", conn, "teachers")
                conn.execute(
                    "UPDATE teachers SET user_id = ? WHERE id = ?",
                    (teacher_user_id, new_teacher["id"]),
                )
            # Mark verification as used
            conn.execute(
                "UPDATE email_verifications SET verified = 1 WHERE id = ?",
                (record["id"],),
            )
            conn.commit()
            session.pop("pending_verify_email", None)
            uid_msg = f" Your User ID is: {teacher_user_id}" if new_teacher else ""
            flash(f"Email verified! Your account is active.{uid_msg} Please log in.", "success")
            return redirect(url_for("teacher_login"))
        except sqlite3.IntegrityError:
            flash("Username or email already registered.", "danger")
            return redirect(url_for("teacher_register"))
        finally:
            conn.close()

    return render_template("auth/verify.html", email=email)


@app.route("/auth/resend-otp", methods=["POST"])
def resend_otp():
    email = session.get("pending_verify_email")
    if not email:
        flash("No pending verification.", "warning")
        return redirect(url_for("teacher_register"))

    conn = get_db_connection()
    record = conn.execute(
        "SELECT * FROM email_verifications WHERE email = ? AND verified = 0 ORDER BY created_at DESC LIMIT 1",
        (email,),
    ).fetchone()

    if not record:
        conn.close()
        flash("No pending registration found. Please register again.", "warning")
        return redirect(url_for("teacher_register"))

    # Generate new OTP and update
    new_otp = generate_otp()
    new_expires = (datetime.now() + timedelta(minutes=10)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    conn.execute(
        "UPDATE email_verifications SET otp_code = ?, expires_at = ? WHERE id = ?",
        (new_otp, new_expires, record["id"]),
    )
    conn.commit()
    conn.close()

    email_body = build_verification_email(record["full_name"], new_otp)
    sent = send_email(email, f"New Verification Code — {APP_NAME}", email_body)

    if sent:
        flash("A new verification code has been sent.", "success")
    else:
        flash("Failed to resend code. Please try again.", "danger")

    return redirect(url_for("verify_email"))


@app.route("/auth/forgot-password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        if not email:
            flash("Please enter your email address.", "danger")
            return render_template("auth/forgot_password.html")

        conn = get_db_connection()
        teacher = conn.execute("SELECT * FROM teachers WHERE email = ?", (email,)).fetchone()
        admin = conn.execute("SELECT * FROM admins WHERE email = ?", (email,)).fetchone()

        if not teacher and not admin:
            conn.close()
            flash("If an account with that email exists, a verification code has been sent.", "info")
            return render_template("auth/forgot_password.html")

        user_id = teacher["id"] if teacher else admin["id"]
        user_type = "teacher" if teacher else "admin"
        user_name = teacher["full_name"] if teacher else admin["full_name"]

        # Generate 5-digit OTP
        otp = generate_otp(5)
        otp_hash = hashlib.sha256(otp.encode()).hexdigest()
        expires_at = (datetime.now() + timedelta(minutes=10)).strftime("%Y-%m-%d %H:%M:%S")

        # Invalidate previous tokens
        conn.execute("UPDATE password_resets SET used = 1 WHERE teacher_id = ? AND user_type = ? AND used = 0", (user_id, user_type))
        conn.execute("INSERT INTO password_resets (teacher_id, token, expires_at, user_type) VALUES (?, ?, ?, ?)", (user_id, otp_hash, expires_at, user_type))
        conn.commit()
        conn.close()

        # Store in session for verification
        session["pw_reset_email"] = email
        session["pw_reset_user_id"] = user_id
        session["pw_reset_user_type"] = user_type

        # Send OTP email
        email_body = build_reset_email(user_name, otp)
        send_email(email, f"Password Reset Code — {APP_NAME}", email_body)

        flash("A 5-digit verification code has been sent to your email.", "info")
        # Mask email for display
        parts = email.split("@")
        masked = parts[0][:2] + "***@" + parts[1] if len(parts) == 2 else email
        return render_template("auth/verify_password_otp.html", masked_email=masked, raw_email=email)

    return render_template("auth/forgot_password.html")


@app.route("/auth/verify-password-otp", methods=["POST"])
def verify_password_otp():
    otp = request.form.get("otp", "").strip()
    email = session.get("pw_reset_email", "")
    user_id = session.get("pw_reset_user_id")
    user_type = session.get("pw_reset_user_type", "teacher")

    if not otp or not user_id:
        flash("Session expired. Please start again.", "danger")
        return redirect(url_for("forgot_password"))

    otp_hash = hashlib.sha256(otp.encode()).hexdigest()
    conn = get_db_connection()
    record = conn.execute(
        "SELECT * FROM password_resets WHERE teacher_id = ? AND user_type = ? AND token = ? AND used = 0",
        (user_id, user_type, otp_hash),
    ).fetchone()

    if not record:
        conn.close()
        flash("Invalid verification code. Please try again.", "danger")
        parts = email.split("@")
        masked = parts[0][:2] + "***@" + parts[1] if len(parts) == 2 else email
        return render_template("auth/verify_password_otp.html", masked_email=masked, raw_email=email)

    expires = datetime.strptime(record["expires_at"], "%Y-%m-%d %H:%M:%S")
    if datetime.now() > expires:
        conn.close()
        flash("Verification code has expired. Please request a new one.", "danger")
        return redirect(url_for("forgot_password"))

    conn.close()

    # OTP verified — allow password reset
    session["pw_reset_verified"] = True
    return render_template("auth/reset_password.html")


@app.route("/auth/reset-password", methods=["GET", "POST"])
def reset_password():
    if not session.get("pw_reset_verified"):
        flash("Please verify your email first.", "danger")
        return redirect(url_for("forgot_password"))

    if request.method == "POST":
        new_password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if len(new_password) < 6:
            flash("Password must be at least 6 characters.", "danger")
            return render_template("auth/reset_password.html")

        if new_password != confirm_password:
            flash("Passwords do not match.", "danger")
            return render_template("auth/reset_password.html")

        user_id = session.get("pw_reset_user_id")
        user_type = session.get("pw_reset_user_type", "teacher")

        new_hash = hashlib.sha256(new_password.encode()).hexdigest()
        table = "admins" if user_type == "admin" else "teachers"
        conn = get_db_connection()
        conn.execute(f"UPDATE {table} SET password_hash = ? WHERE id = ?", (new_hash, user_id))
        conn.execute("UPDATE password_resets SET used = 1 WHERE teacher_id = ? AND user_type = ? AND used = 0", (user_id, user_type))
        conn.commit()
        conn.close()

        # Clear session
        for k in ("pw_reset_email", "pw_reset_user_id", "pw_reset_user_type", "pw_reset_verified"):
            session.pop(k, None)

        flash("Password reset successfully! Please log in.", "success")
        return redirect(url_for("admin_login"))

    return render_template("auth/reset_password.html")


# ── Forgot PIN flow (OTP-based) ──
@app.route("/auth/forgot-pin", methods=["GET", "POST"])
def forgot_pin():
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        if not email:
            flash("Please enter your email address.", "danger")
            return render_template("auth/forgot_pin.html")

        conn = get_db_connection()
        teacher = conn.execute("SELECT * FROM teachers WHERE email = ?", (email,)).fetchone()
        admin = conn.execute("SELECT * FROM admins WHERE email = ?", (email,)).fetchone()

        if not teacher and not admin:
            conn.close()
            flash("If an account with that email exists, a verification code has been sent.", "info")
            return render_template("auth/forgot_pin.html")

        user_id = teacher["id"] if teacher else admin["id"]
        user_type = "teacher" if teacher else "admin"
        user_name = teacher["full_name"] if teacher else admin["full_name"]

        # Generate 5-digit OTP
        otp = generate_otp(5)
        otp_hash = hashlib.sha256(otp.encode()).hexdigest()
        expires_at = (datetime.now() + timedelta(minutes=10)).strftime("%Y-%m-%d %H:%M:%S")

        conn.execute("UPDATE password_resets SET used = 1 WHERE teacher_id = ? AND user_type = ? AND used = 0", (user_id, user_type))
        conn.execute("INSERT INTO password_resets (teacher_id, token, expires_at, user_type) VALUES (?, ?, ?, ?)", (user_id, otp_hash, expires_at, user_type))
        conn.commit()
        conn.close()

        session["pin_reset_email"] = email
        session["pin_reset_user_id"] = user_id
        session["pin_reset_user_type"] = user_type

        email_body = build_pin_reset_email(user_name, otp)
        send_email(email, f"PIN Reset Code — {APP_NAME}", email_body)

        flash("A 5-digit verification code has been sent to your email.", "info")
        parts = email.split("@")
        masked = parts[0][:2] + "***@" + parts[1] if len(parts) == 2 else email
        return render_template("auth/verify_pin_otp.html", masked_email=masked, raw_email=email)

    return render_template("auth/forgot_pin.html")


@app.route("/auth/verify-pin-otp", methods=["POST"])
def verify_pin_otp():
    otp = request.form.get("otp", "").strip()
    email = session.get("pin_reset_email", "")
    user_id = session.get("pin_reset_user_id")
    user_type = session.get("pin_reset_user_type", "teacher")

    if not otp or not user_id:
        flash("Session expired. Please start again.", "danger")
        return redirect(url_for("forgot_pin"))

    otp_hash = hashlib.sha256(otp.encode()).hexdigest()
    conn = get_db_connection()
    record = conn.execute(
        "SELECT * FROM password_resets WHERE teacher_id = ? AND user_type = ? AND token = ? AND used = 0",
        (user_id, user_type, otp_hash),
    ).fetchone()

    if not record:
        conn.close()
        flash("Invalid verification code. Please try again.", "danger")
        parts = email.split("@")
        masked = parts[0][:2] + "***@" + parts[1] if len(parts) == 2 else email
        return render_template("auth/verify_pin_otp.html", masked_email=masked, raw_email=email)

    expires = datetime.strptime(record["expires_at"], "%Y-%m-%d %H:%M:%S")
    if datetime.now() > expires:
        conn.close()
        flash("Verification code has expired. Please request a new one.", "danger")
        return redirect(url_for("forgot_pin"))

    conn.close()

    session["pin_reset_verified"] = True
    return render_template("auth/reset_pin.html")


@app.route("/auth/reset-pin", methods=["GET", "POST"])
def reset_pin():
    if not session.get("pin_reset_verified"):
        flash("Please verify your email first.", "danger")
        return redirect(url_for("forgot_pin"))

    if request.method == "POST":
        new_pin = request.form.get("new_pin", "").strip()
        confirm_pin = request.form.get("confirm_pin", "").strip()

        if not new_pin.isdigit() or len(new_pin) < 4 or len(new_pin) > 6:
            flash("PIN must be 4-6 digits.", "danger")
            return render_template("auth/reset_pin.html")

        if new_pin != confirm_pin:
            flash("PINs do not match.", "danger")
            return render_template("auth/reset_pin.html")

        user_id = session.get("pin_reset_user_id")
        user_type = session.get("pin_reset_user_type", "teacher")

        new_hash = hashlib.sha256(new_pin.encode()).hexdigest()
        table = "admins" if user_type == "admin" else "teachers"
        conn = get_db_connection()
        conn.execute(f"UPDATE {table} SET user_pin = ? WHERE id = ?", (new_hash, user_id))
        conn.execute("UPDATE password_resets SET used = 1 WHERE teacher_id = ? AND user_type = ? AND used = 0", (user_id, user_type))
        conn.commit()
        conn.close()

        for k in ("pin_reset_email", "pin_reset_user_id", "pin_reset_user_type", "pin_reset_verified"):
            session.pop(k, None)

        flash("PIN reset successfully! Please log in.", "success")
        return redirect(url_for("teacher_login"))

    return render_template("auth/reset_pin.html")


@app.route("/auth/pin", methods=["GET", "POST"])
def teacher_login():
    # If already logged in, redirect to appropriate dashboard
    if "admin_id" in session:
        return redirect(url_for("admin_dashboard"))
    if "teacher_id" in session:
        return redirect(url_for("teacher_dashboard"))

    # Get pending admin from email+password step
    pending_admin_id = session.get("pending_pin_admin_id")

    if request.method == "POST":
        user_id_input = request.form.get("user_id", "").strip()
        password = request.form.get("password", "")
        remember = request.form.get("remember") == "1"

        if not user_id_input or not password:
            flash("Please enter both User ID and User PIN.", "danger")
            return render_template("auth/login.html")

        # Hash the PIN for comparison
        pin_hash = hashlib.sha256(password.encode()).hexdigest()

        conn = get_db_connection()
        teacher = conn.execute(
            "SELECT * FROM teachers WHERE user_id = ?", (user_id_input,)
        ).fetchone()
        conn.close()

        if teacher:
            # If admin did email+password first, only allow their own teachers/staff
            if pending_admin_id:
                teacher_admin_id = teacher["registered_by_admin_id"] if "registered_by_admin_id" in teacher.keys() else 0
                if teacher_admin_id and teacher_admin_id != pending_admin_id:
                    flash("This user does not belong to your admin account.", "danger")
                    return render_template("auth/login.html")

            # Check user_pin first, fall back to password_hash for legacy accounts
            stored_pin = teacher["user_pin"] if "user_pin" in teacher.keys() else ""
            if (stored_pin and stored_pin == pin_hash) or (not stored_pin and teacher["password_hash"] == pin_hash):
                # Check if email is verified (for new accounts)
                is_verified = teacher["is_verified"] if "is_verified" in teacher.keys() else 1
                if not is_verified:
                    flash("Please verify your email before logging in.", "warning")
                    return render_template("auth/login.html")

                # Clear pending admin state
                session.pop("pending_pin_admin_id", None)
                session.pop("pending_pin_admin_name", None)
                session.pop("pending_pin_admin_username", None)
                session.pop("pending_pin_admin_image", None)
                session.pop("pending_pin_remember", None)

                # Create session
                session["teacher_id"] = teacher["id"]
                session["teacher_name"] = teacher["full_name"]
                session["teacher_username"] = teacher["username"]
                session["teacher_profile_image"] = teacher["profile_image"] if "profile_image" in teacher.keys() else ""
                session.permanent = remember

                flash(f'Welcome back, {teacher["full_name"]}!', "success")
                return redirect(url_for("teacher_dashboard"))
            else:
                flash("Invalid User ID or User PIN.", "danger")
        else:
            # Check admins table by user_id + user_pin
            conn = get_db_connection()
            admin = conn.execute(
                "SELECT * FROM admins WHERE user_id = ?", (user_id_input,)
            ).fetchone()
            conn.close()

            if admin:
                # If admin did email+password first, only allow their own admin account
                if pending_admin_id and admin["id"] != pending_admin_id:
                    flash("This admin account does not match your login.", "danger")
                    return render_template("auth/login.html")

                stored_pin = admin["user_pin"] if "user_pin" in admin.keys() else ""
                if stored_pin and stored_pin == pin_hash:
                    # Clear any pending state
                    session.pop("pending_pin_admin_id", None)
                    session.pop("pending_pin_admin_name", None)
                    session.pop("pending_pin_admin_username", None)
                    session.pop("pending_pin_admin_image", None)
                    pending_remember = session.pop("pending_pin_remember", None)

                    session["admin_id"] = admin["id"]
                    session["admin_name"] = admin["full_name"]
                    session["admin_username"] = admin["username"]
                    session["admin_profile_image"] = admin["profile_image"] if "profile_image" in admin.keys() else ""
                    session.permanent = pending_remember if pending_remember is not None else remember

                    flash(f'Welcome back, {admin["full_name"]}!', "success")
                    return redirect(url_for("admin_dashboard"))
                else:
                    flash("Invalid User ID or User PIN.", "danger")
            else:
                flash("Invalid User ID or User PIN.", "danger")

    return render_template("auth/login.html")


@app.route("/auth/logout", methods=["GET", "POST"])
def teacher_logout():
    session.pop("teacher_id", None)
    session.pop("teacher_name", None)
    session.pop("teacher_username", None)
    session.pop("teacher_profile_image", None)
    flash("You have been signed out successfully.", "info")
    return redirect(url_for("teacher_login"))


# ═══════════════════════════════════════════
# Admin Auth Routes (College Gate Portal)
# ═══════════════════════════════════════════
@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if "admin_id" in session:
        return redirect(url_for("admin_dashboard"))

    if request.method == "POST":
        email_input = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        remember = request.form.get("remember") == "1"

        if not email_input or not password:
            flash("Please enter both Email ID and password.", "danger")
            return render_template("auth/admin_login.html")

        password_hash = hashlib.sha256(password.encode()).hexdigest()

        conn = get_db_connection()
        admin = conn.execute(
            "SELECT * FROM admins WHERE email = ?", (email_input,)
        ).fetchone()
        conn.close()

        if admin and admin["password_hash"] == password_hash:
            # Don't set admin_id yet — store pending state for PIN verification
            session["pending_pin_admin_id"] = admin["id"]
            session["pending_pin_admin_name"] = admin["full_name"]
            session["pending_pin_admin_username"] = admin["username"]
            session["pending_pin_admin_image"] = admin["profile_image"] if "profile_image" in admin.keys() else ""
            session["pending_pin_remember"] = remember

            flash(f'Welcome, {admin["full_name"]}! Now verify with your User ID & PIN.', "success")
            return redirect(url_for("teacher_login"))
        else:
            flash("Invalid email or password.", "danger")

    return render_template("auth/admin_login.html")


# ── Admin Create Teacher Account ──
@app.route("/api/admin/create_teacher", methods=["POST"])
@admin_login_required
def admin_create_teacher():
    """Admin creates a teacher/staff account directly (no email verification needed).
    Admin sets a 4-6 digit User PIN, or it is auto-generated. Emails credentials to the teacher/staff."""
    data = request.get_json()
    full_name = (data.get("full_name") or "").strip()
    email = (data.get("email") or "").strip()
    phone = (data.get("phone") or "").strip()
    department = (data.get("department") or "").strip()
    designation = (data.get("designation") or "").strip()
    custom_pin = (data.get("user_pin") or "").strip()

    if not full_name or not email:
        return jsonify({"success": False, "message": "Full name and email are required"}), 400
    if "@" not in email:
        return jsonify({"success": False, "message": "Valid email is required"}), 400

    # Validate custom PIN if provided
    if custom_pin:
        if not custom_pin.isdigit() or len(custom_pin) < 4 or len(custom_pin) > 6:
            return jsonify({"success": False, "message": "PIN must be 4-6 digits"}), 400
        user_pin_plain = custom_pin
    else:
        user_pin_plain = generate_otp(6)

    username = email.split("@")[0].lower().replace(" ", "").replace(".", "_")

    user_pin_hash = hashlib.sha256(user_pin_plain.encode()).hexdigest()

    conn = get_db_connection()
    existing = conn.execute(
        "SELECT id FROM teachers WHERE username = ? OR email = ?", (username, email)
    ).fetchone()
    if existing:
        conn.close()
        return jsonify({"success": False, "message": "A teacher/staff with this email already exists"}), 409

    try:
        conn.execute(
            """INSERT INTO teachers (username, password_hash, email, full_name, phone, department, designation, is_verified, user_pin, registered_by_admin_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?)""",
            (username, user_pin_hash, email, full_name, phone, department, designation, user_pin_hash, session.get("admin_id", 0)),
        )
        new_teacher = conn.execute("SELECT id FROM teachers WHERE username = ?", (username,)).fetchone()
        teacher_user_id = ""
        if new_teacher:
            teacher_user_id = generate_user_id("TCH", conn, "teachers")
            conn.execute("UPDATE teachers SET user_id = ? WHERE id = ?", (teacher_user_id, new_teacher["id"]))
        conn.commit()
        conn.close()

        # Send credentials email to teacher/staff with ALL details
        role_label = designation if designation in ("HOD", "Staff") else "Teacher"
        admin_id = session.get("admin_id", 0)
        admin_row = get_db_connection().execute("SELECT full_name, email FROM admins WHERE id = ?", (admin_id,)).fetchone()
        adm_name = admin_row["full_name"] if admin_row else ""
        adm_email = admin_row["email"] if admin_row else ""
        cred_body = build_credentials_email(
            full_name, teacher_user_id, user_pin_plain, role_label,
            email=email, department=department, designation=designation,
            phone=phone, admin_name=adm_name, admin_email=adm_email,
        )
        send_email(email, f"Your {APP_NAME} Account — Login Credentials", cred_body)

        return jsonify({
            "success": True,
            "message": f"Account created for {full_name}. Credentials sent to {email}.",
            "user_id": teacher_user_id,
            "teacher": {
                "name": full_name, "email": email, "user_id": teacher_user_id,
                "phone": phone, "department": department, "designation": designation,
            },
        })
    except sqlite3.IntegrityError:
        conn.close()
        return jsonify({"success": False, "message": "Username or email already exists"}), 409


@app.route("/admin/register", methods=["GET", "POST"])
def admin_register():
    if "admin_id" in session:
        return redirect(url_for("admin_dashboard"))

    if request.method == "POST":
        password = request.form.get("password", "")
        email = request.form.get("email", "").strip()
        full_name = request.form.get("full_name", "").strip()
        college_name = request.form.get("college_name", "").strip()
        phone = request.form.get("phone", "").strip()
        institution_type = request.form.get("institution_type", "college").strip()
        user_pin = request.form.get("user_pin", "").strip()

        # Auto-generate username from email
        username = email.split("@")[0].lower().replace(" ", "").replace(".", "_")

        if not password or not email or not full_name or not user_pin:
            flash("All required fields must be filled.", "danger")
            return render_template("auth/admin_register.html")

        if len(password) < 6:
            flash("Password must be at least 6 characters.", "danger")
            return render_template("auth/admin_register.html")

        if not user_pin.isdigit() or len(user_pin) < 4 or len(user_pin) > 6:
            flash("User PIN must be 4-6 digits.", "danger")
            return render_template("auth/admin_register.html")

        conn = get_db_connection()
        existing = conn.execute(
            "SELECT id FROM admins WHERE username = ? OR email = ?",
            (username, email),
        ).fetchone()
        if existing:
            conn.close()
            flash("Username or email already exists.", "danger")
            return render_template("auth/admin_register.html")

        password_hash = hashlib.sha256(password.encode()).hexdigest()

        # Generate OTP
        otp_code = generate_otp()
        expires_at = (datetime.now() + timedelta(minutes=10)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        # Delete any previous pending verifications for the same email
        conn.execute("DELETE FROM email_verifications WHERE email = ? AND user_type = 'admin'", (email,))

        # Store pending registration
        conn.execute(
            """INSERT INTO email_verifications
               (email, username, password_hash, full_name, otp_code, expires_at, user_type, college_name, phone, institution_type)
               VALUES (?, ?, ?, ?, ?, ?, 'admin', ?, ?, ?)""",
            (email, username, password_hash, full_name, otp_code, expires_at, college_name, phone, institution_type),
        )
        conn.commit()
        conn.close()

        # Send verification email (admin pink theme)
        email_body = build_admin_verification_email(full_name, otp_code)
        sent = send_email(email, f"Admin Verification — {APP_NAME}", email_body)

        if sent:
            session["pending_admin_verify_email"] = email
            session["pending_admin_password"] = password
            session["pending_admin_pin"] = user_pin
            flash("A verification code has been sent to your email.", "info")
            return redirect(url_for("admin_verify_email"))
        else:
            flash(
                "Failed to send verification email. Please try again.", "danger"
            )
            return render_template("auth/admin_register.html")

    return render_template("auth/admin_register.html")


@app.route("/admin/verify", methods=["GET", "POST"])
def admin_verify_email():
    email = session.get("pending_admin_verify_email")
    if not email:
        flash("No pending verification. Please register first.", "warning")
        return redirect(url_for("admin_register"))

    if request.method == "POST":
        otp_input = request.form.get("otp", "").strip()

        if not otp_input:
            flash("Please enter the verification code.", "danger")
            return render_template("auth/admin_verify.html", email=email)

        conn = get_db_connection()
        record = conn.execute(
            """SELECT * FROM email_verifications
               WHERE email = ? AND otp_code = ? AND verified = 0 AND user_type = 'admin'
               ORDER BY created_at DESC LIMIT 1""",
            (email, otp_input),
        ).fetchone()

        if not record:
            conn.close()
            flash("Invalid verification code.", "danger")
            return render_template("auth/admin_verify.html", email=email)

        # Check expiry
        expires = datetime.strptime(record["expires_at"], "%Y-%m-%d %H:%M:%S")
        if datetime.now() > expires:
            conn.close()
            flash("Verification code has expired. Please register again.", "danger")
            return redirect(url_for("admin_register"))

        # OTP valid — create the admin account
        try:
            inst_type = ""
            try:
                inst_type = record["institution_type"] or "college"
            except (IndexError, KeyError):
                inst_type = "college"
            conn.execute(
                """INSERT INTO admins (username, password_hash, email, full_name, college_name, phone, institution_type)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    record["username"],
                    record["password_hash"],
                    record["email"],
                    record["full_name"],
                    record["college_name"] or "",
                    record["phone"] or "",
                    inst_type,
                ),
            )
            # Auto-generate user_id for the new admin
            new_admin = conn.execute(
                "SELECT id FROM admins WHERE username = ?", (record["username"],)
            ).fetchone()
            if new_admin:
                admin_user_id = generate_user_id("ADM", conn, "admins")
                # Use the PIN set by admin during registration
                admin_pin_plain = session.get("pending_admin_pin", "") or generate_otp(6)
                admin_pin_hash = hashlib.sha256(admin_pin_plain.encode()).hexdigest()
                conn.execute(
                    "UPDATE admins SET user_id = ?, user_pin = ? WHERE id = ?",
                    (admin_user_id, admin_pin_hash, new_admin["id"]),
                )
            conn.execute(
                "UPDATE email_verifications SET verified = 1 WHERE id = ?",
                (record["id"],),
            )
            conn.commit()
            # Send welcome email with credentials (email ID + password + User ID + PIN)
            plain_password = session.pop("pending_admin_password", "")
            session.pop("pending_admin_pin", None)
            if plain_password:
                welcome_body = build_welcome_email(record["full_name"], record["email"], plain_password, "Admin",
                                                   user_id=admin_user_id if new_admin else "",
                                                   user_pin=admin_pin_plain if new_admin else "")
                send_email(record["email"], f"Welcome to {APP_NAME} — Your Admin Credentials", welcome_body)
            session.pop("pending_admin_verify_email", None)
            uid_msg = f" Your User ID is: {admin_user_id}" if new_admin else ""
            flash(f"Email verified! Your admin account is active.{uid_msg} Please sign in.", "success")
            return redirect(url_for("admin_login"))
        except sqlite3.IntegrityError:
            flash("Username or email already registered.", "danger")
            return redirect(url_for("admin_register"))
        finally:
            conn.close()

    return render_template("auth/admin_verify.html", email=email)


@app.route("/admin/resend-otp", methods=["POST"])
def admin_resend_otp():
    email = session.get("pending_admin_verify_email")
    if not email:
        flash("No pending verification.", "warning")
        return redirect(url_for("admin_register"))

    conn = get_db_connection()
    record = conn.execute(
        "SELECT * FROM email_verifications WHERE email = ? AND verified = 0 AND user_type = 'admin' ORDER BY created_at DESC LIMIT 1",
        (email,),
    ).fetchone()

    if not record:
        conn.close()
        flash("No pending registration found. Please register again.", "warning")
        return redirect(url_for("admin_register"))

    # Generate new OTP and update
    new_otp = generate_otp()
    new_expires = (datetime.now() + timedelta(minutes=10)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    conn.execute(
        "UPDATE email_verifications SET otp_code = ?, expires_at = ? WHERE id = ?",
        (new_otp, new_expires, record["id"]),
    )
    conn.commit()
    conn.close()

    email_body = build_admin_verification_email(record["full_name"], new_otp)
    sent = send_email(email, f"New Admin Verification Code — {APP_NAME}", email_body)

    if sent:
        flash("A new verification code has been sent.", "success")
    else:
        flash("Failed to resend code. Please try again.", "danger")

    return redirect(url_for("admin_verify_email"))


@app.route("/admin/logout", methods=["GET", "POST"])
def admin_logout():
    session.pop("admin_id", None)
    session.pop("admin_name", None)
    session.pop("admin_username", None)
    session.pop("admin_profile_image", None)
    flash("Admin signed out successfully.", "info")
    return redirect(url_for("admin_login"))


# ═══════════════════════════════════════════
# Admin Dashboard Routes (College Gate Portal)
# ═══════════════════════════════════════════
@app.route("/admin/dashboard")
@admin_login_required
def admin_dashboard():
    admin_id = session["admin_id"]
    conn = get_db_connection()
    today = datetime.now().strftime("%d-%m-%Y")

    # Only users registered by THIS admin
    all_users = conn.execute(
        "SELECT * FROM users WHERE registered_by_admin_id = ? ORDER BY name",
        (admin_id,)
    ).fetchall()

    # Today's gate attendance recorded by THIS admin
    gate_attendance = conn.execute(
        "SELECT ar.*, u.user_id, u.department, u.email, u.role, u.phone "
        "FROM attendance_records ar "
        "LEFT JOIN users u ON ar.student_name = u.username "
        "WHERE ar.date = ? AND ar.attendance_type = 'gate' AND ar.admin_id = ? "
        "ORDER BY ar.time DESC",
        (today, admin_id)
    ).fetchall()

    # Today's class attendance (from all teachers — class data is teacher-managed)
    class_attendance = conn.execute(
        "SELECT ar.*, u.user_id, u.department, u.email, u.role, c.name as class_name, t.full_name as teacher_name "
        "FROM attendance_records ar "
        "LEFT JOIN users u ON ar.student_name = u.username "
        "LEFT JOIN classes c ON ar.class_id = c.id "
        "LEFT JOIN teachers t ON ar.teacher_id = t.id "
        "WHERE ar.date = ? AND ar.attendance_type = 'class' "
        "ORDER BY ar.time DESC",
        (today,)
    ).fetchall()

    # Stats
    total_users = len(all_users)
    total_students = len([u for u in all_users if u["role"] == "Student"])
    total_teachers_count = conn.execute("SELECT COUNT(*) FROM teachers").fetchone()[0]
    total_staff = len([u for u in all_users if u["role"] in ("Staff", "Admin")])
    gate_count = len(gate_attendance)
    class_count = len(class_attendance)
    
    # Total admins
    total_admins = conn.execute("SELECT COUNT(*) FROM admins").fetchone()[0]
    
    # Week attendance stats (last 7 days) — gate filtered by this admin
    week_stats = []
    for i in range(6, -1, -1):
        day = (datetime.now() - timedelta(days=i)).strftime("%d-%m-%Y")
        day_label = (datetime.now() - timedelta(days=i)).strftime("%a")
        day_gate = conn.execute(
            "SELECT COUNT(*) FROM attendance_records WHERE date = ? AND attendance_type = 'gate' AND admin_id = ?", (day, admin_id)
        ).fetchone()[0]
        day_class = conn.execute(
            "SELECT COUNT(*) FROM attendance_records WHERE date = ? AND attendance_type = 'class'", (day,)
        ).fetchone()[0]
        week_stats.append({"label": day_label, "date": day, "gate": day_gate, "class": day_class})
    
    # Total all-time records by this admin (gate) + all class records
    total_records = conn.execute(
        "SELECT COUNT(*) FROM attendance_records WHERE (attendance_type = 'gate' AND admin_id = ?) OR attendance_type = 'class'",
        (admin_id,)
    ).fetchone()[0]

    # Department-wise gate attendance
    dept_stats = {}
    for rec in gate_attendance:
        dept = rec["department"] or "Unknown"
        dept_stats[dept] = dept_stats.get(dept, 0) + 1
    
    # Role-wise user breakdown
    role_stats = {}
    for u in all_users:
        role = u["role"] or "Unknown"
        role_stats[role] = role_stats.get(role, 0) + 1

    conn.close()

    return render_template(
        "admin/dashboard.html",
        all_users=all_users,
        gate_attendance=gate_attendance,
        class_attendance=class_attendance,
        total_users=total_users,
        total_students=total_students,
        total_teachers=total_teachers_count,
        total_staff=total_staff,
        total_admins=total_admins,
        gate_count=gate_count,
        class_count=class_count,
        total_records=total_records,
        dept_stats=dept_stats,
        role_stats=role_stats,
        week_stats=week_stats,
        today=today,
    )


@app.route("/api/admin/stats")
@admin_login_required
def admin_stats_api():
    conn = get_db_connection()
    today = datetime.now().strftime("%d-%m-%Y")

    total_users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    gate_today = conn.execute(
        "SELECT COUNT(*) FROM attendance_records WHERE date = ? AND attendance_type = 'gate'", (today,)
    ).fetchone()[0]
    class_today = conn.execute(
        "SELECT COUNT(*) FROM attendance_records WHERE date = ? AND attendance_type = 'class'", (today,)
    ).fetchone()[0]
    total_teachers = conn.execute("SELECT COUNT(*) FROM teachers").fetchone()[0]

    conn.close()
    return jsonify({
        "total_users": total_users,
        "gate_today": gate_today,
        "class_today": class_today,
        "total_teachers": total_teachers,
    })


@app.route("/admin/attendance")
@admin_login_required
def admin_attendance():
    conn = get_db_connection()
    today = datetime.now().strftime("%d-%m-%Y")

    # Default to today's records
    records = conn.execute(
        "SELECT ar.*, u.user_id, u.department, u.email, u.role, u.phone, "
        "u.registered_by, c.name as class_name, t.full_name as teacher_name "
        "FROM attendance_records ar "
        "LEFT JOIN users u ON ar.student_name = u.username "
        "LEFT JOIN classes c ON ar.class_id = c.id "
        "LEFT JOIN teachers t ON ar.teacher_id = t.id "
        "WHERE ar.date = ? "
        "ORDER BY ar.time DESC",
        (today,)
    ).fetchall()

    departments = conn.execute(
        "SELECT DISTINCT department FROM users WHERE department IS NOT NULL AND department != '' ORDER BY department"
    ).fetchall()
    departments = [d['department'] for d in departments]

    conn.close()
    today_html = datetime.now().strftime("%Y-%m-%d")
    return render_template("admin/attendance.html", records=records, today=today, today_html=today_html, departments=departments)


@app.route("/admin/attendance/filter", methods=["POST"])
@admin_login_required
def admin_filter_attendance():
    data = request.get_json() or {}
    date_filter = data.get("date", "")
    att_type = data.get("type", "")
    dept_filter = data.get("department", "")
    role_filter = data.get("role", "")

    query = (
        "SELECT ar.*, u.user_id, u.department, u.email, u.role, u.phone, "
        "u.registered_by, c.name as class_name, t.full_name as teacher_name "
        "FROM attendance_records ar "
        "LEFT JOIN users u ON ar.student_name = u.username "
        "LEFT JOIN classes c ON ar.class_id = c.id "
        "LEFT JOIN teachers t ON ar.teacher_id = t.id WHERE 1=1"
    )
    params = []

    if date_filter:
        query += " AND ar.date = ?"
        params.append(date_filter)
    if att_type:
        query += " AND ar.attendance_type = ?"
        params.append(att_type)
    if dept_filter:
        query += " AND u.department = ?"
        params.append(dept_filter)
    if role_filter:
        query += " AND u.role = ?"
        params.append(role_filter)

    query += " ORDER BY ar.date DESC, ar.time DESC LIMIT 500"

    conn = get_db_connection()
    records = conn.execute(query, params).fetchall()
    conn.close()

    return jsonify({
        "success": True,
        "attendance": [
            {
                "id": r["id"],
                "student_name": r["student_name"],
                "date": r["date"],
                "time": r["time"],
                "status": r["status"],
                "attendance_type": r["attendance_type"] if "attendance_type" in r.keys() else "gate",
                "user_id": r["user_id"] or "",
                "department": r["department"] or "",
                "email": r["email"] or "",
                "role": r["role"] or "",
                "phone": r["phone"] or "",
                "registered_by": r["registered_by"] or "",
                "class_name": r["class_name"] or "",
                "teacher_name": r["teacher_name"] or "",
            }
            for r in records
        ],
    })


# Teacher dashboard routes
@app.route("/teacher/dashboard")
@teacher_login_required
def teacher_dashboard():
    teacher_id = session["teacher_id"]
    conn = get_db_connection()

    # Get classes for this teacher
    classes = conn.execute(
        "SELECT * FROM classes WHERE teacher_id = ?", (teacher_id,)
    ).fetchall()

    # Get total students count
    students_count = conn.execute(
        """SELECT COUNT(DISTINCT cs.student_name) as count
           FROM class_students cs
           JOIN classes c ON cs.class_id = c.id
           WHERE c.teacher_id = ?""",
        (teacher_id,)
    ).fetchone()['count']

    # Get recent attendance records (today's records) with student details
    today = datetime.now().strftime("%d-%m-%Y")
    recent_attendance = conn.execute(
        """SELECT ar.*, c.name as class_name,
                  u.user_id as student_user_id, u.department as student_department,
                  u.email as student_email, u.role as student_role, u.phone as student_phone
           FROM attendance_records ar 
           LEFT JOIN classes c ON ar.class_id = c.id 
           LEFT JOIN users u ON ar.student_name = u.username
           WHERE ar.teacher_id = ? AND ar.date = ?
           ORDER BY ar.time DESC LIMIT 10""",
        (teacher_id, today),
    ).fetchall()

    # Get today's attendance rate
    today_attendance_count = len(recent_attendance)
    attendance_rate = round((today_attendance_count / students_count) * 100, 1) if students_count > 0 else 0

    # Get total attendance records for this teacher
    total_records = conn.execute(
        "SELECT COUNT(*) as count FROM attendance_records WHERE teacher_id = ?",
        (teacher_id,)
    ).fetchone()['count']

    # Get per-class attendance stats for today
    class_stats = conn.execute(
        """SELECT c.name, COUNT(ar.id) as count
           FROM classes c
           LEFT JOIN attendance_records ar ON ar.class_id = c.id AND ar.date = ?
           WHERE c.teacher_id = ?
           GROUP BY c.id ORDER BY count DESC""",
        (today, teacher_id)
    ).fetchall()

    # Get weekly attendance trend (last 7 days)
    week_stats = []
    for i in range(6, -1, -1):
        d = datetime.now() - timedelta(days=i)
        dstr = d.strftime("%d-%m-%Y")
        cnt = conn.execute(
            "SELECT COUNT(*) as count FROM attendance_records WHERE teacher_id = ? AND date = ?",
            (teacher_id, dstr)
        ).fetchone()['count']
        week_stats.append({"day": d.strftime("%a"), "date": dstr, "count": cnt})

    conn.close()

    return render_template(
        "teacher/dashboard.html", 
        classes=classes, 
        recent_attendance=recent_attendance,
        students_count=students_count,
        attendance_rate=attendance_rate,
        total_records=total_records,
        class_stats=class_stats,
        week_stats=week_stats
    )


# ═══════════════════════════════════════════
# Teacher Profile Routes
# ═══════════════════════════════════════════
@app.route("/teacher/profile", methods=["GET", "POST"])
@teacher_login_required
def teacher_profile():
    teacher_id = session["teacher_id"]
    conn = get_db_connection()

    if request.method == "POST":
        full_name = request.form.get("full_name", "").strip()
        email = request.form.get("email", "").strip()
        phone = request.form.get("phone", "").strip()
        bio = request.form.get("bio", "").strip()

        # Department & Designation are admin-controlled — keep existing values
        teacher_row = conn.execute("SELECT department, designation FROM teachers WHERE id = ?", (teacher_id,)).fetchone()
        department = teacher_row["department"] if teacher_row else ""
        designation = teacher_row["designation"] if teacher_row else ""

        if not full_name or not email:
            flash("Name and email are required.", "danger")
            teacher = conn.execute("SELECT * FROM teachers WHERE id = ?", (teacher_id,)).fetchone()
            conn.close()
            return render_template("teacher/profile.html", teacher=teacher)

        # Check email uniqueness (exclude self)
        existing = conn.execute(
            "SELECT id FROM teachers WHERE email = ? AND id != ?", (email, teacher_id)
        ).fetchone()
        if existing:
            flash("This email is already used by another account.", "danger")
            teacher = conn.execute("SELECT * FROM teachers WHERE id = ?", (teacher_id,)).fetchone()
            conn.close()
            return render_template("teacher/profile.html", teacher=teacher)

        # Handle profile image upload
        profile_image_url = None
        if "profile_image" in request.files:
            file = request.files["profile_image"]
            if file and file.filename:
                profile_image_url = upload_profile_image(file, folder="smart_attendance/teachers")
                if profile_image_url:
                    conn.execute("UPDATE teachers SET profile_image = ? WHERE id = ?", (profile_image_url, teacher_id))

        conn.execute(
            """UPDATE teachers SET full_name = ?, email = ?, phone = ?, department = ?,
               designation = ?, bio = ? WHERE id = ?""",
            (full_name, email, phone, department, designation, bio, teacher_id),
        )
        conn.commit()

        # Update session
        session["teacher_name"] = full_name
        if profile_image_url:
            session["teacher_profile_image"] = profile_image_url

        flash("Profile updated successfully!", "success")
        conn.close()
        return redirect(url_for("teacher_profile"))

    teacher = conn.execute("SELECT * FROM teachers WHERE id = ?", (teacher_id,)).fetchone()
    conn.close()
    return render_template("teacher/profile.html", teacher=teacher)


@app.route("/teacher/profile/remove-image", methods=["POST"])
@teacher_login_required
def teacher_remove_image():
    teacher_id = session["teacher_id"]
    conn = get_db_connection()
    conn.execute("UPDATE teachers SET profile_image = '' WHERE id = ?", (teacher_id,))
    conn.commit()
    conn.close()
    session["teacher_profile_image"] = ""
    flash("Profile image removed.", "info")
    return redirect(url_for("teacher_profile"))


@app.route("/api/teacher/contact_admin", methods=["POST"])
@teacher_login_required
def teacher_contact_admin():
    """Teacher sends a help request email to their admin."""
    data = request.get_json()
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"success": False, "message": "Message is required"}), 400

    teacher_id = session["teacher_id"]
    conn = get_db_connection()
    teacher = conn.execute("SELECT * FROM teachers WHERE id = ?", (teacher_id,)).fetchone()
    if not teacher:
        conn.close()
        return jsonify({"success": False, "message": "Teacher not found"}), 404

    # Find the admin who registered this teacher
    admin_id = teacher["registered_by_admin_id"] if "registered_by_admin_id" in teacher.keys() else 0
    admin = None
    if admin_id:
        admin = conn.execute("SELECT full_name, email FROM admins WHERE id = ?", (admin_id,)).fetchone()
    if not admin:
        # Fallback: find any admin
        admin = conn.execute("SELECT full_name, email FROM admins ORDER BY id LIMIT 1").fetchone()
    conn.close()

    if not admin:
        return jsonify({"success": False, "message": "No admin found to contact"}), 404

    admin_email_addr = admin["email"]
    teacher_name = teacher["full_name"]
    teacher_email = teacher["email"]
    teacher_dept = teacher["department"] if "department" in teacher.keys() else ""
    teacher_desig = teacher["designation"] if "designation" in teacher.keys() else ""
    teacher_uid = teacher["user_id"] if "user_id" in teacher.keys() else ""

    email_body = f"""
    <div style="font-family:'Segoe UI',Arial,sans-serif;max-width:560px;margin:0 auto;background:#fff;border-radius:16px;overflow:hidden;box-shadow:0 4px 24px rgba(0,0,0,.1);">
      <div style="background:linear-gradient(135deg,#667eea,#764ba2);padding:2rem;text-align:center;color:#fff;">
        <div style="width:60px;height:60px;background:rgba(255,255,255,.2);border-radius:50%;margin:0 auto 1rem;display:flex;align-items:center;justify-content:center;font-size:1.6rem;">&#128172;</div>
        <h2 style="margin:0;font-size:1.4rem;">Help Request from Teacher</h2>
        <p style="opacity:.85;margin:.4rem 0 0;font-size:.9rem;">Smart Attendance System</p>
      </div>
      <div style="padding:2rem;">
        <p style="color:#555;font-size:.95rem;">Hello <strong>{admin['full_name']}</strong>,</p>
        <p style="color:#555;font-size:.95rem;">A teacher has sent you a request:</p>
        <div style="background:#f8f9ff;border:1px solid #e8ecff;border-radius:12px;padding:1.2rem;margin:1.2rem 0;">
          <table style="width:100%;border-collapse:collapse;">
            <tr><td style="padding:6px 12px;color:#888;font-size:.85rem;font-weight:600;">Name</td>
                <td style="padding:6px 12px;font-weight:700;color:#333;">{teacher_name}</td></tr>
            <tr><td style="padding:6px 12px;color:#888;font-size:.85rem;font-weight:600;border-top:1px solid #eef;">Email</td>
                <td style="padding:6px 12px;color:#333;border-top:1px solid #eef;">{teacher_email}</td></tr>
            <tr><td style="padding:6px 12px;color:#888;font-size:.85rem;font-weight:600;border-top:1px solid #eef;">User ID</td>
                <td style="padding:6px 12px;color:#333;border-top:1px solid #eef;">{teacher_uid}</td></tr>
            <tr><td style="padding:6px 12px;color:#888;font-size:.85rem;font-weight:600;border-top:1px solid #eef;">Department</td>
                <td style="padding:6px 12px;color:#333;border-top:1px solid #eef;">{teacher_dept}</td></tr>
            <tr><td style="padding:6px 12px;color:#888;font-size:.85rem;font-weight:600;border-top:1px solid #eef;">Designation</td>
                <td style="padding:6px 12px;color:#333;border-top:1px solid #eef;">{teacher_desig}</td></tr>
          </table>
        </div>
        <div style="background:#fff8e1;border:1px solid #ffe082;border-radius:10px;padding:1rem;margin:1rem 0;">
          <p style="color:#333;font-size:.92rem;margin:0;"><strong>Message:</strong></p>
          <p style="color:#555;font-size:.92rem;margin:.4rem 0 0;">{message}</p>
        </div>
        <p style="color:#888;font-size:.82rem;">You can reply directly to this teacher at <a href="mailto:{teacher_email}" style="color:#667eea;">{teacher_email}</a></p>
      </div>
      <div style="background:#f8f8ff;padding:1rem;text-align:center;font-size:.75rem;color:#aaa;border-top:1px solid #eee;">Smart Attendance System &copy; 2025</div>
    </div>
    """
    sent = send_email(admin_email_addr, f"Help Request from {teacher_name} — {APP_NAME}", email_body)
    if sent:
        return jsonify({"success": True, "message": f"Your request has been sent to the admin ({admin['full_name']}). They will contact you via email."})
    else:
        return jsonify({"success": False, "message": "Failed to send email. Please try again later."}), 500


# ═══════════════════════════════════════════
# Admin Profile Routes
# ═══════════════════════════════════════════
@app.route("/admin/profile", methods=["GET", "POST"])
@admin_login_required
def admin_profile():
    admin_id = session["admin_id"]
    conn = get_db_connection()

    if request.method == "POST":
        full_name = request.form.get("full_name", "").strip()
        email = request.form.get("email", "").strip()
        phone = request.form.get("phone", "").strip()
        college_name = request.form.get("college_name", "").strip()
        designation = request.form.get("designation", "").strip()
        bio = request.form.get("bio", "").strip()
        institution_type = request.form.get("institution_type", "").strip()

        if not full_name or not email:
            flash("Name and email are required.", "danger")
            admin = conn.execute("SELECT * FROM admins WHERE id = ?", (admin_id,)).fetchone()
            conn.close()
            return render_template("admin/profile.html", admin=admin)

        # Check email uniqueness (exclude self)
        existing = conn.execute(
            "SELECT id FROM admins WHERE email = ? AND id != ?", (email, admin_id)
        ).fetchone()
        if existing:
            flash("This email is already used by another account.", "danger")
            admin = conn.execute("SELECT * FROM admins WHERE id = ?", (admin_id,)).fetchone()
            conn.close()
            return render_template("admin/profile.html", admin=admin)

        # Handle profile image upload
        profile_image_url = None
        if "profile_image" in request.files:
            file = request.files["profile_image"]
            if file and file.filename:
                profile_image_url = upload_profile_image(file, folder="smart_attendance/admins")
                if profile_image_url:
                    conn.execute("UPDATE admins SET profile_image = ? WHERE id = ?", (profile_image_url, admin_id))

        conn.execute(
            """UPDATE admins SET full_name = ?, email = ?, phone = ?, college_name = ?,
               designation = ?, bio = ?, institution_type = ? WHERE id = ?""",
            (full_name, email, phone, college_name, designation, bio, institution_type, admin_id),
        )
        conn.commit()

        # Update session
        session["admin_name"] = full_name
        if profile_image_url:
            session["admin_profile_image"] = profile_image_url

        flash("Profile updated successfully!", "success")
        conn.close()
        return redirect(url_for("admin_profile"))

    admin = conn.execute("SELECT * FROM admins WHERE id = ?", (admin_id,)).fetchone()
    conn.close()
    return render_template("admin/profile.html", admin=admin)


@app.route("/admin/profile/remove-image", methods=["POST"])
@admin_login_required
def admin_remove_image():
    admin_id = session["admin_id"]
    conn = get_db_connection()
    conn.execute("UPDATE admins SET profile_image = '' WHERE id = ?", (admin_id,))
    conn.commit()
    conn.close()
    session["admin_profile_image"] = ""
    flash("Profile image removed.", "info")
    return redirect(url_for("admin_profile"))


# API endpoint to get student count
@app.route("/api/teacher/stats")
@teacher_login_required
def teacher_stats():
    teacher_id = session["teacher_id"]
    conn = get_db_connection()

    # Get total students
    students_count = conn.execute(
        """SELECT COUNT(DISTINCT cs.student_name) as count
           FROM class_students cs
           JOIN classes c ON cs.class_id = c.id
           WHERE c.teacher_id = ?""",
        (teacher_id,)
    ).fetchone()['count']

    # Get today's attendance count
    today = datetime.now().strftime("%d-%m-%Y")
    today_attendance = conn.execute(
        """SELECT COUNT(*) as count
           FROM attendance_records
           WHERE teacher_id = ? AND date = ?""",
        (teacher_id, today)
    ).fetchone()['count']

    # Calculate attendance rate
    attendance_rate = 0
    if students_count > 0:
        attendance_rate = round((today_attendance / students_count) * 100, 1)

    conn.close()

    return jsonify({
        "success": True,
        "students_count": students_count,
        "today_attendance": today_attendance,
        "attendance_rate": attendance_rate
    })


# Class management routes
@app.route("/teacher/classes", methods=["GET", "POST"])
@teacher_login_required
def manage_classes():
    if request.method == "POST":
        name = request.form["name"]
        description = request.form.get("description", "")
        department = request.form.get("department", "")
        teacher_id = session["teacher_id"]

        conn = get_db_connection()
        
        # Check if department column exists, if not add it
        try:
            conn.execute("SELECT department FROM classes LIMIT 1")
        except:
            conn.execute("ALTER TABLE classes ADD COLUMN department TEXT")
            conn.commit()
        
        conn.execute(
            "INSERT INTO classes (teacher_id, name, description, department) VALUES (?, ?, ?, ?)",
            (teacher_id, name, description, department),
        )
        conn.commit()
        conn.close()

        flash("Class created successfully!", "success")
        return redirect(url_for("manage_classes"))

    conn = get_db_connection()
    
    # Ensure department column exists
    try:
        conn.execute("SELECT department FROM classes LIMIT 1")
    except:
        conn.execute("ALTER TABLE classes ADD COLUMN department TEXT")
        conn.commit()
    
    classes = conn.execute(
        "SELECT * FROM classes WHERE teacher_id = ?", (session["teacher_id"],)
    ).fetchall()
    
    # Get student counts and today's attendance for each class
    today = datetime.now().strftime("%d-%m-%Y")
    classes_with_counts = []
    for class_row in classes:
        student_count = conn.execute(
            "SELECT COUNT(*) as count FROM class_students WHERE class_id = ?",
            (class_row["id"],)
        ).fetchone()["count"]
        
        today_att = conn.execute(
            "SELECT COUNT(*) as count FROM attendance_records WHERE class_id = ? AND date = ?",
            (class_row["id"], today)
        ).fetchone()["count"]
        
        class_dict = dict(class_row)
        class_dict["student_count"] = student_count
        class_dict["today_attendance"] = today_att
        classes_with_counts.append(class_dict)
    
    conn.close()

    return render_template("teacher/classes.html", classes=classes_with_counts)


@app.route("/teacher/edit_class/<int:class_id>", methods=["POST"])
@teacher_login_required
def edit_class(class_id):
    """Edit class name, department, description via AJAX."""
    conn = get_db_connection()
    class_info = conn.execute(
        "SELECT * FROM classes WHERE id = ? AND teacher_id = ?",
        (class_id, session["teacher_id"]),
    ).fetchone()

    if not class_info:
        conn.close()
        return jsonify({"success": False, "message": "Class not found or access denied"}), 404

    if request.is_json:
        data = request.get_json()
    else:
        data = request.form

    name = data.get("name", "").strip()
    department = data.get("department", "").strip()
    description = data.get("description", "").strip()

    if not name:
        conn.close()
        return jsonify({"success": False, "message": "Class name is required"}), 400

    conn.execute(
        "UPDATE classes SET name = ?, department = ?, description = ? WHERE id = ? AND teacher_id = ?",
        (name, department, description, class_id, session["teacher_id"]),
    )
    conn.commit()
    conn.close()

    return jsonify({"success": True, "message": "Class updated successfully"})


@app.route("/teacher/class/<int:class_id>")
@teacher_login_required
def view_class(class_id):
    conn = get_db_connection()
    class_info = conn.execute(
        "SELECT * FROM classes WHERE id = ? AND teacher_id = ?",
        (class_id, session["teacher_id"]),
    ).fetchone()

    if not class_info:
        conn.close()
        flash("Class not found or access denied.", "danger")
        return redirect(url_for("manage_classes"))

    # Get students in this class with user details
    students = conn.execute(
        """SELECT cs.student_name, cs.added_at,
                  u.user_id, u.department, u.email, u.role, u.phone, u.name as full_name
           FROM class_students cs
           LEFT JOIN users u ON cs.student_name = u.username
           WHERE cs.class_id = ? ORDER BY cs.student_name""",
        (class_id,)
    ).fetchall()

    # Get attendance records with user details
    attendance_records = conn.execute(
        """SELECT a.id, a.student_name, a.class_id, a.teacher_id, a.date, a.time, a.status,
                  u.user_id as student_user_id, u.department as student_department, u.email as student_email
           FROM attendance_records a
           LEFT JOIN users u ON a.student_name = u.username
           WHERE a.class_id = ? AND a.teacher_id = ?
           ORDER BY a.date DESC, a.time DESC LIMIT 50""",
        (class_id, session["teacher_id"]),
    ).fetchall()

    # Get all registered users (for adding students dropdown)
    all_registered = conn.execute(
        "SELECT username FROM users ORDER BY username"
    ).fetchall()

    conn.close()

    # Serialize attendance for JSON
    attendance = [
        {
            "id": record["id"],
            "student_name": record["student_name"],
            "class_id": record["class_id"],
            "teacher_id": record["teacher_id"],
            "date": str(record["date"]),
            "time": str(record["time"]),
            "status": record["status"],
            "student_user_id": record["student_user_id"] or "",
            "student_department": record["student_department"] or "",
            "student_email": record["student_email"] or "",
        }
        for record in attendance_records
    ]

    return render_template(
        "teacher/class_detail.html", 
        class_info=class_info, 
        attendance=attendance,
        students=students,
        all_registered=all_registered
    )


@app.route("/teacher/add_student_to_class/<int:class_id>", methods=["POST"])
@teacher_login_required
def add_student_to_class(class_id):
    student_name = request.form.get("student_name", "").strip()
    if not student_name:
        return (
            jsonify({"success": False, "message": "Student name cannot be empty"}),
            400,
        )

    conn = get_db_connection()
    try:
        # Verify class ownership
        class_info = conn.execute(
            "SELECT * FROM classes WHERE id = ? AND teacher_id = ?",
            (class_id, session["teacher_id"]),
        ).fetchone()

        if not class_info:
            return (
                jsonify(
                    {"success": False, "message": "Class not found or access denied"}
                ),
                404,
            )

        # Check if student exists in face data
        if os.path.exists("data/names.pkl"):
            with open("data/names.pkl", "rb") as f:
                existing_students = pickle.load(f)
            if student_name not in existing_students:
                return (
                    jsonify(
                        {
                            "success": False,
                            "message": f'Student "{student_name}" needs face registration first',
                        }
                    ),
                    400,
                )

        # Add to class
        conn.execute(
            "INSERT INTO class_students (class_id, student_name) VALUES (?, ?)",
            (class_id, student_name),
        )
        conn.commit()
        return jsonify(
            {
                "success": True,
                "message": f"{student_name} added to class successfully",
                "student": student_name,
            }
        )

    except sqlite3.IntegrityError:
        return (
            jsonify(
                {
                    "success": False,
                    "message": f"{student_name} is already in this class",
                }
            ),
            409,
        )
    except Exception as e:
        print(f"Error adding student: {str(e)}")
        return jsonify({"success": False, "message": "Database error occurred"}), 500
    finally:
        conn.close()


@app.route("/teacher/remove_student_from_class/<int:class_id>", methods=["POST"])
@teacher_login_required
def remove_student_from_class(class_id):
    student_name = request.form.get("student_name", "").strip()
    if not student_name:
        return jsonify({"success": False, "message": "Student name required"}), 400

    conn = get_db_connection()
    try:
        # Verify class ownership
        class_info = conn.execute(
            "SELECT * FROM classes WHERE id = ? AND teacher_id = ?",
            (class_id, session["teacher_id"]),
        ).fetchone()

        if not class_info:
            return jsonify({"success": False, "message": "Class not found or access denied"}), 404

        # Remove student from class
        result = conn.execute(
            "DELETE FROM class_students WHERE class_id = ? AND student_name = ?",
            (class_id, student_name),
        )
        conn.commit()

        if result.rowcount > 0:
            return jsonify({
                "success": True,
                "message": f"{student_name} removed from class successfully"
            })
        else:
            return jsonify({
                "success": False,
                "message": f"{student_name} not found in this class"
            }), 404

    except Exception as e:
        print(f"Error removing student: {str(e)}")
        return jsonify({"success": False, "message": "Database error occurred"}), 500
    finally:
        conn.close()


@app.route("/teacher/delete_class/<int:class_id>", methods=["POST"])
@teacher_login_required
def delete_class(class_id):
    conn = get_db_connection()
    # Verify class belongs to teacher
    class_info = conn.execute(
        "SELECT * FROM classes WHERE id = ? AND teacher_id = ?",
        (class_id, session["teacher_id"]),
    ).fetchone()

    if not class_info:
        conn.close()
        flash("Class not found or access denied.", "danger")
        return redirect(url_for("manage_classes"))

    # Delete class and associated records
    conn.execute("DELETE FROM class_students WHERE class_id = ?", (class_id,))
    conn.execute("DELETE FROM attendance_records WHERE class_id = ?", (class_id,))
    conn.execute("DELETE FROM classes WHERE id = ?", (class_id,))
    conn.commit()
    conn.close()

    flash("Class deleted successfully.", "success")
    return redirect(url_for("manage_classes"))


# Enhanced attendance views
@app.route("/teacher/attendance")
@teacher_login_required
def teacher_attendance():
    teacher_id = session["teacher_id"]
    conn = get_db_connection()

    # Get all classes for filter dropdown
    classes = conn.execute(
        "SELECT * FROM classes WHERE teacher_id = ?", (teacher_id,)
    ).fetchall()

    # Get all students for this teacher
    students_query = """
    SELECT DISTINCT cs.student_name 
    FROM class_students cs
    JOIN classes c ON cs.class_id = c.id
    WHERE c.teacher_id = ?
    """
    students = conn.execute(students_query, (teacher_id,)).fetchall()

    # Default to showing today's attendance with user details
    today = datetime.now().strftime("%d-%m-%Y")

    attendance_query = """
    SELECT ar.*, c.name as class_name,
           u.user_id as student_user_id, u.department as student_department,
           u.email as student_email, u.role as student_role, u.phone as student_phone
    FROM attendance_records ar
    LEFT JOIN classes c ON ar.class_id = c.id
    LEFT JOIN users u ON ar.student_name = u.username
    WHERE ar.teacher_id = ? AND ar.date = ?
    ORDER BY ar.time DESC
    """
    attendance = conn.execute(attendance_query, (teacher_id, today)).fetchall()

    conn.close()

    return render_template(
        "teacher/attendance.html",
        classes=classes,
        students=students,
        attendance=attendance,
        selected_date=datetime.now().strftime("%Y-%m-%d"),
    )


@app.route("/teacher/class/<int:class_id>/student_attendance")
@teacher_login_required
def class_student_attendance(class_id):
    teacher_id = session["teacher_id"]
    conn = get_db_connection()

    # Verify class belongs to teacher
    class_info = conn.execute(
        "SELECT * FROM classes WHERE id = ? AND teacher_id = ?", (class_id, teacher_id)
    ).fetchone()

    if not class_info:
        conn.close()
        flash("Class not found or access denied.", "danger")
        return redirect(url_for("manage_classes"))

    # Get all students in this class with user details
    students = conn.execute(
        """SELECT cs.student_name, cs.added_at,
                  u.user_id, u.department, u.email, u.role, u.phone
           FROM class_students cs
           LEFT JOIN users u ON cs.student_name = u.username
           WHERE cs.class_id = ?""",
        (class_id,)
    ).fetchall()

    # Get student names
    student_names = [student["student_name"] for student in students]

    # Get attendance records for all students in this class
    attendance_records = {}
    attendance_dates = set()

    for student_name in student_names:
        query = """
        SELECT * FROM attendance_records 
        WHERE class_id = ? AND teacher_id = ? AND student_name = ?
        ORDER BY date ASC, time ASC
        """
        student_records = conn.execute(
            query, (class_id, teacher_id, student_name)
        ).fetchall()

        attendance_records[student_name] = {}
        for record in student_records:
            date = record["date"]
            attendance_records[student_name][date] = record
            attendance_dates.add(date)

    # Sort dates
    sorted_dates = sorted(list(attendance_dates))

    # Generate attendance summary with user details
    attendance_summary = []
    for s in students:
        student_name = s["student_name"]
        present_count = sum(
            1 for date in sorted_dates if date in attendance_records.get(student_name, {})
        )
        attendance_rate = round(present_count / len(sorted_dates) * 100, 1) if sorted_dates else 0

        student_summary = {
            "name": student_name,
            "user_id": s["user_id"] or "",
            "department": s["department"] or "",
            "email": s["email"] or "",
            "role": s["role"] or "",
            "phone": s["phone"] or "",
            "present_count": present_count,
            "total_days": len(sorted_dates),
            "attendance_rate": attendance_rate,
        }
        attendance_summary.append(student_summary)

    conn.close()

    return render_template(
        "teacher/student_attendance.html",
        class_info=class_info,
        students=students,
        attendance_records=attendance_records,
        dates=sorted_dates,
        attendance_summary=attendance_summary,
    )


@app.route("/teacher/class/<int:class_id>/student_attendance/filter", methods=["POST"])
@teacher_login_required
def filter_class_student_attendance(class_id):
    start_date = request.form.get("start_date")
    end_date = request.form.get("end_date")
    student_name = request.form.get("student_name")

    conn = get_db_connection()

    # Verify class belongs to teacher
    class_info = conn.execute(
        "SELECT * FROM classes WHERE id = ? AND teacher_id = ?",
        (class_id, session["teacher_id"]),
    ).fetchone()

    if not class_info:
        conn.close()
        return (
            jsonify({"success": False, "message": "Class not found or access denied"}),
            403,
        )

    # Base query for student attendance
    query = """
    SELECT * FROM attendance_records 
    WHERE class_id = ? AND teacher_id = ?
    """
    params = [class_id, session["teacher_id"]]

    # Add date range filter if provided
    if start_date and end_date:
        query += " AND date BETWEEN ? AND ?"
        params.extend([start_date, end_date])

    # Add student filter if provided
    if student_name:
        query += " AND student_name = ?"
        params.append(student_name)

    query += " ORDER BY date ASC, time ASC"

    filtered_records = conn.execute(query, params).fetchall()

    # Get unique dates and students
    dates = set()
    students = set()

    for record in filtered_records:
        dates.add(record["date"])
        students.add(record["student_name"])

    # Sort dates
    sorted_dates = sorted(list(dates))
    sorted_students = sorted(list(students))

    # Organize by student and date
    attendance_records = {}
    for student_name in sorted_students:
        attendance_records[student_name] = {}

    for record in filtered_records:
        student_name = record["student_name"]
        date = record["date"]
        attendance_records[student_name][date] = record

    # Generate attendance summary
    attendance_summary = []
    for student_name in sorted_students:
        present_count = sum(
            1 for date in sorted_dates if date in attendance_records[student_name]
        )
        attendance_rate = present_count / len(sorted_dates) * 100 if sorted_dates else 0

        student_summary = {
            "name": student_name,
            "present_count": present_count,
            "total_days": len(sorted_dates),
            "attendance_rate": round(attendance_rate, 1),
        }
        attendance_summary.append(student_summary)

    # Convert to JSON-serializable format
    result = {
        "success": True,
        "dates": sorted_dates,
        "students": sorted_students,
        "attendance_records": {
            student: {
                date: dict(attendance_records[student][date])
                for date in attendance_records[student]
            }
            for student in attendance_records
        },
        "attendance_summary": attendance_summary,
    }

    conn.close()
    return jsonify(result)


@app.route("/teacher/attendance/filter", methods=["POST"])
@teacher_login_required
def filter_attendance():
    # Handle both form data and JSON
    if request.is_json:
        data = request.get_json()
        date_filter = data.get("date")
        class_filter = data.get("class_id")
        student_filter = data.get("student_name")
    else:
        date_filter = request.form.get("date")
        class_filter = request.form.get("class_id")
        student_filter = request.form.get("student_name")

    query = """SELECT ar.*, c.name as class_name,
                      u.user_id as student_user_id, u.department as student_department,
                      u.email as student_email, u.role as student_role, u.phone as student_phone
               FROM attendance_records ar 
               LEFT JOIN classes c ON ar.class_id = c.id 
               LEFT JOIN users u ON ar.student_name = u.username
               WHERE ar.teacher_id = ?"""
    params = [session["teacher_id"]]

    if date_filter:
        # Convert YYYY-MM-DD from HTML input to DD-MM-YYYY for database
        try:
            date_obj = datetime.strptime(date_filter, "%Y-%m-%d")
            date_filter = date_obj.strftime("%d-%m-%Y")
        except:
            pass  # Already in DD-MM-YYYY or other format
        query += " AND ar.date = ?"
        params.append(date_filter)

    if class_filter:
        # Accept both class ID (int) and class name (string)
        try:
            class_id_int = int(class_filter)
            query += " AND ar.class_id = ?"
            params.append(class_id_int)
        except (ValueError, TypeError):
            # It's a class name, look up the ID
            query += " AND c.name = ?"
            params.append(class_filter)

    if student_filter:
        query += " AND ar.student_name LIKE ?"
        params.append(f"%{student_filter}%")

    query += " ORDER BY ar.date DESC, ar.time DESC"

    conn = get_db_connection()
    attendance = conn.execute(query, params).fetchall()
    conn.close()

    # Convert to list of dicts for JSON response
    result = []
    for row in attendance:
        d = dict(row)
        d['class_name'] = d.get('class_name') or '—'
        d['student_user_id'] = d.get('student_user_id') or ''
        d['student_department'] = d.get('student_department') or ''
        d['student_email'] = d.get('student_email') or ''
        d['student_role'] = d.get('student_role') or ''
        d['student_phone'] = d.get('student_phone') or ''
        result.append(d)
    return jsonify({"success": True, "attendance": result})


@app.route("/teacher/export_today")
@teacher_login_required
def export_today_attendance():
    # Get today's date
    today = datetime.now().strftime("%d-%m-%Y")

    # Query for today's attendance records
    query = """
    SELECT ar.*, c.name as class_name 
    FROM attendance_records ar 
    LEFT JOIN classes c ON ar.class_id = c.id 
    WHERE ar.teacher_id = ? AND ar.date = ? 
    ORDER BY ar.class_id, ar.student_name, ar.time
    """

    conn = get_db_connection()
    attendance = conn.execute(query, (session["teacher_id"], today)).fetchall()
    conn.close()

    if not attendance:
        flash("No attendance records for today.", "warning")
        return redirect(url_for("teacher_dashboard"))

    # Create CSV file
    filename = f"today_attendance_{today}.csv"
    filepath = os.path.join("data", filename)

    with open(filepath, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Student Name", "Class", "Time", "Status"])

        for record in attendance:
            writer.writerow(
                [
                    record["student_name"],
                    record["class_name"] or "No Class",
                    record["time"],
                    record["status"],
                ]
            )

    flash("Today's attendance exported successfully!", "success")
    return send_from_directory("data", filename, as_attachment=True)


@app.route("/teacher/export", methods=["POST"])
@teacher_login_required
def teacher_export_attendance():
    # Handle both form data and JSON
    if request.is_json:
        data = request.get_json()
        date_filter = data.get("date")
        class_filter = data.get("class_id")
        student_filter = data.get("student_name")
    else:
        date_filter = request.form.get("date")
        class_filter = request.form.get("class_id")
        student_filter = request.form.get("student_name")

    query = """SELECT ar.*, c.name as class_name,
                      u.user_id as student_user_id, u.department as student_department, u.email as student_email
               FROM attendance_records ar 
               LEFT JOIN classes c ON ar.class_id = c.id 
               LEFT JOIN users u ON ar.student_name = u.username
               WHERE ar.teacher_id = ?"""
    params = [session["teacher_id"]]

    if date_filter:
        query += " AND ar.date = ?"
        params.append(date_filter)

    if class_filter:
        query += " AND ar.class_id = ?"
        params.append(int(class_filter))

    if student_filter:
        query += " AND ar.student_name = ?"
        params.append(student_filter)

    query += " ORDER BY ar.date DESC, ar.time DESC"

    conn = get_db_connection()
    attendance = conn.execute(query, params).fetchall()
    conn.close()

    if not attendance:
        return jsonify({"success": False, "message": "No attendance data to export"}), 404

    # Create CSV file in memory
    from io import StringIO
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["Student Name", "User ID", "Department", "Email", "Class", "Date", "Time", "Status", "Notes"])

    for record in attendance:
        writer.writerow(
            [
                record["student_name"],
                record.get("student_user_id") or "",
                record.get("student_department") or "",
                record.get("student_email") or "",
                record["class_name"] or "No Class",
                record["date"],
                record["time"],
                record["status"],
                record.get("notes", "") or "",
            ]
        )

    # Create response
    output.seek(0)
    filename = f'attendance_{datetime.now().strftime("%d-%m-%Y_%H-%M-%S")}.csv'
    
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment;filename={filename}"}
    )


# Delete attendance record endpoint
@app.route("/teacher/delete_record/<int:record_id>", methods=["DELETE", "POST"])
@teacher_login_required
def delete_attendance_record(record_id):
    teacher_id = session["teacher_id"]
    conn = get_db_connection()

    # Verify the record belongs to this teacher
    record = conn.execute(
        "SELECT * FROM attendance_records WHERE id = ? AND teacher_id = ?",
        (record_id, teacher_id)
    ).fetchone()

    if not record:
        conn.close()
        return jsonify({"success": False, "message": "Record not found or access denied"}), 404

    # Delete the record
    conn.execute("DELETE FROM attendance_records WHERE id = ?", (record_id,))
    conn.commit()
    conn.close()

    return jsonify({"success": True, "message": "Record deleted successfully"})


# Store attendance in database when recognition happens
def store_attendance_in_db(student_name, date, time_str, class_id=None):
    print(f"\n*** STORING ATTENDANCE: {student_name}, {date}, {time_str} ***")
    try:
        # Set default values
        status = "Present"
        teacher_id = None
        admin_id = None
        attendance_type = "gate"  # default

        # Determine who is logged in and set attendance_type accordingly
        if "admin_id" in session and "teacher_id" not in session:
            admin_id = session["admin_id"]
            attendance_type = "gate"
            print(f"Using admin ID {admin_id} from session — gate attendance")
        elif "teacher_id" in session:
            teacher_id = session["teacher_id"]
            attendance_type = "class"
            print(f"Using teacher ID {teacher_id} from session — class attendance")

            # Look for class_id if not provided and teacher is logged in
            if class_id is None and student_name:
                conn = get_db_connection()
                # Try to find a class this student belongs to
                class_result = conn.execute(
                    """SELECT class_id FROM class_students 
                       WHERE student_name = ? AND class_id IN 
                       (SELECT id FROM classes WHERE teacher_id = ?)""",
                    (student_name, teacher_id),
                ).fetchone()

                if class_result:
                    class_id = class_result["class_id"]
                    print(f"Found class {class_id} for student {student_name}")
                conn.close()
        else:
            # If no one is logged in, use a default teacher ID of 1
            teacher_id = 1
            print(
                f"No user logged in, using default teacher ID 1 for {student_name}"
            )

        # Make sure we have a valid teacher_id
        if not teacher_id:
            teacher_id = 1
            print(f"Teacher ID was None, using default value 1")

        # Insert attendance record
        conn = get_db_connection()

        # Check if there are any existing teachers in the database
        teacher_check = conn.execute(
            "SELECT COUNT(*) as count FROM teachers"
        ).fetchone()
        print(f"Found {teacher_check['count']} teachers in database")

        # If no teacher exists and we're using the default ID, create a default teacher
        if teacher_check["count"] == 0 and teacher_id == 1:
            try:
                conn.execute(
                    "INSERT INTO teachers (id, username, password_hash, email, full_name) VALUES (?, ?, ?, ?, ?)",
                    (
                        1,
                        "default",
                        "default_hash",
                        "default@example.com",
                        "Default Teacher",
                    ),
                )
                conn.commit()
                print("Created default teacher for attendance records")
            except sqlite3.IntegrityError as e:
                print(
                    f"Error creating default teacher: {str(e)}, but continuing anyway"
                )

        # Debug the SQL command we're about to execute
        print(
            f"Executing INSERT with values: ({student_name}, {class_id}, {teacher_id}, {admin_id}, {date}, {time_str}, {status}, {attendance_type})"
        )

        # Insert the attendance record with proper attendance_type and admin_id
        conn.execute(
            "INSERT INTO attendance_records (student_name, class_id, teacher_id, admin_id, date, time, status, attendance_type) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (student_name, class_id, teacher_id, admin_id, date, time_str, status, attendance_type),
        )
        conn.commit()

        # Verify the record was inserted
        verify = conn.execute(
            "SELECT id FROM attendance_records WHERE student_name = ? AND date = ? AND time = ?",
            (student_name, date, time_str),
        ).fetchone()

        conn.close()

        if verify:
            print(
                f"✓ Successfully stored attendance record #{verify['id']} for {student_name} in database"
            )
            return True
        else:
            print(f"✗ Failed to verify attendance record was stored for {student_name}")
            return False

    except Exception as e:
        print(f"ERROR STORING ATTENDANCE: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


# Note: We're using the original recognize route defined above
# This comment replaces the duplicate route


# Test route to manually create an attendance record for debugging
@app.route("/test_attendance")
def test_attendance():
    student_name = "Test Student"
    date = datetime.now().strftime("%d-%m-%Y")
    time_str = datetime.now().strftime("%H:%M:%S")

    print(f"\n********* CREATING TEST ATTENDANCE RECORD *********")
    success = store_attendance_in_db(student_name, date, time_str, None)

    # Also check database records
    conn = get_db_connection()
    records = conn.execute("SELECT * FROM attendance_records").fetchall()
    conn.close()

    info = f"Test record created: {success}\n"
    info += f"Total records in database: {len(records)}\n"

    for record in records:
        info += f"Record #{record['id']}: {record['student_name']} on {record['date']} at {record['time']}\n"

    return render_template("result.html", result=info)


if __name__ == "__main__":
    app.run(debug=True)
