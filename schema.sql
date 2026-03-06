-- Teachers table for authentication
CREATE TABLE IF NOT EXISTS teachers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    full_name TEXT NOT NULL,
    phone TEXT DEFAULT '',
    department TEXT DEFAULT '',
    designation TEXT DEFAULT '',
    bio TEXT DEFAULT '',
    profile_image TEXT DEFAULT '',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Admins table for college gate / admin portal
CREATE TABLE IF NOT EXISTS admins (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    full_name TEXT NOT NULL,
    college_name TEXT DEFAULT '',
    phone TEXT DEFAULT '',
    designation TEXT DEFAULT '',
    bio TEXT DEFAULT '',
    profile_image TEXT DEFAULT '',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Classes table to organize students
CREATE TABLE IF NOT EXISTS classes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    teacher_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    department TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (teacher_id) REFERENCES teachers (id)
);

-- Student-class mapping
CREATE TABLE IF NOT EXISTS class_students (
    class_id INTEGER NOT NULL,
    student_name TEXT NOT NULL,
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (class_id, student_name),
    FOREIGN KEY (class_id) REFERENCES classes (id)
);

-- Enhanced attendance records with teacher/admin association
-- attendance_type: 'gate' (admin scans at college gate) or 'class' (teacher scans in classroom)
CREATE TABLE IF NOT EXISTS attendance_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_name TEXT NOT NULL,
    class_id INTEGER,
    teacher_id INTEGER,
    admin_id INTEGER,
    date TEXT NOT NULL,
    time TEXT NOT NULL,
    status TEXT NOT NULL,
    attendance_type TEXT NOT NULL DEFAULT 'gate',
    notes TEXT,
    FOREIGN KEY (class_id) REFERENCES classes (id),
    FOREIGN KEY (teacher_id) REFERENCES teachers (id),
    FOREIGN KEY (admin_id) REFERENCES admins (id)
);

-- Registered users (students/teachers/staff) from face capture
CREATE TABLE IF NOT EXISTS users (
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
    registered_by_role TEXT DEFAULT ''
);

-- Global classes managed by admin (for school & college)
CREATE TABLE IF NOT EXISTS admin_classes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    institution_type TEXT NOT NULL DEFAULT 'college',
    department TEXT DEFAULT '',
    description TEXT DEFAULT '',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Assign admin-managed classes to teachers/HODs
CREATE TABLE IF NOT EXISTS teacher_class_assignments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    admin_class_id INTEGER NOT NULL,
    teacher_id INTEGER NOT NULL,
    assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(admin_class_id, teacher_id),
    FOREIGN KEY (admin_class_id) REFERENCES admin_classes (id) ON DELETE CASCADE,
    FOREIGN KEY (teacher_id) REFERENCES teachers (id) ON DELETE CASCADE
);
