from flask import Flask, render_template, request, redirect, session, jsonify, send_from_directory

import sqlite3
import os
import base64
import cv2
import numpy as np

from werkzeug.utils import secure_filename


# =========================================================
# APP CONFIGURATION
# =========================================================

app = Flask(__name__)

app.secret_key = "ai-securetrack-secret-key"

UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# =========================================================
# FACE RECOGNITION VARIABLES
# =========================================================

face_recognizer = cv2.face.LBPHFaceRecognizer_create()

face_names = {}

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


# =========================================================
# DATABASE CONNECTION
# =========================================================

def get_db_connection():

    conn = sqlite3.connect("database.db")

    conn.row_factory = sqlite3.Row

    return conn


# =========================================================
# DATABASE INITIALIZATION
# =========================================================

def init_db():

    conn = get_db_connection()

    # -----------------------------------------------------
    # USERS TABLE
    # -----------------------------------------------------

    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT DEFAULT 'admin'
        )
    """)

    # -----------------------------------------------------
    # PERSONS TABLE
    # -----------------------------------------------------

    conn.execute("""
        CREATE TABLE IF NOT EXISTS persons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            person_id TEXT,
            description TEXT,
            employee_id TEXT,
            phone TEXT,
            department TEXT,
            photo TEXT,
            status TEXT DEFAULT 'Active'
        )
    """)

    # -----------------------------------------------------
    # VEHICLES TABLE
    # -----------------------------------------------------

    conn.execute("""
        CREATE TABLE IF NOT EXISTS vehicles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            vehicle_number TEXT NOT NULL,
            owner_name TEXT,
            brand TEXT,
            model TEXT,
            colour TEXT,
            status TEXT DEFAULT 'Active'
        )
    """)

    # -----------------------------------------------------
    # DETECTIONS TABLE
    # -----------------------------------------------------

    conn.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            object_type TEXT,
            name TEXT,
            status TEXT,
            confidence REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # -----------------------------------------------------
    # ALERTS TABLE
    # -----------------------------------------------------

    conn.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            alert_type TEXT,
            message TEXT,
            status TEXT DEFAULT 'Active',
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # -----------------------------------------------------
    # TRACKING TABLE
    # -----------------------------------------------------

    conn.execute("""
        CREATE TABLE IF NOT EXISTS tracking (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            status TEXT,
            camera TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # -----------------------------------------------------
    # CREATE DEFAULT ADMIN
    # -----------------------------------------------------

    existing_user = conn.execute(
        "SELECT * FROM users WHERE username = ?",
        ("admin",)
    ).fetchone()

    if existing_user is None:

        conn.execute(
            """
            INSERT INTO users
            (username, password, role)
            VALUES (?, ?, ?)
            """,
            ("admin", "admin123", "admin")
        )

    conn.commit()

    conn.close()


# =========================================================
# TRAIN FACE RECOGNIZER
# =========================================================

def train_face_recognizer():

    global face_names

    face_names = {}

    faces = []
    labels = []

    conn = get_db_connection()

    persons = conn.execute(
        "SELECT * FROM persons WHERE photo IS NOT NULL"
    ).fetchall()

    conn.close()

    current_label = 0

    for person in persons:

        photo_filename = person["photo"]

        if not photo_filename:
            continue

        photo_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            photo_filename
        )

        if not os.path.exists(photo_path):
            continue

        image = cv2.imread(photo_path)

        if image is None:
            continue

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        detected_faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50)
        )

        if len(detected_faces) == 0:
            continue

        face_names[current_label] = person["name"]

        for (x, y, w, h) in detected_faces:

            face_image = gray[y:y+h, x:x+w]

            # Resize face to a fixed size for LBPH
            face_image = cv2.resize(face_image, (200, 200))

            faces.append(face_image)

            labels.append(current_label)

        current_label += 1

    if len(faces) > 0:

        face_recognizer.train(
            faces,
            np.array(labels)
        )


# =========================================================
# LOGIN
# =========================================================

@app.route("/", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        username = request.form.get("username")

        password = request.form.get("password")

        conn = get_db_connection()

        user = conn.execute(
            """
            SELECT * FROM users
            WHERE username = ? AND password = ?
            """,
            (username, password)
        ).fetchone()

        conn.close()

        if user:

            session["username"] = user["username"]

            session["role"] = user["role"]

            return redirect("/dashboard")

        return render_template(
            "login.html",
            error="Invalid username or password"
        )

    return render_template("login.html")


# =========================================================
# DASHBOARD
# =========================================================

@app.route("/dashboard")
def dashboard():

    if "username" not in session:
        return redirect("/")

    conn = get_db_connection()

    # Count known persons
    person_count = conn.execute(
        "SELECT COUNT(*) FROM persons"
    ).fetchone()[0]

    # Count known vehicles
    vehicle_count = conn.execute(
        "SELECT COUNT(*) FROM vehicles"
    ).fetchone()[0]

    # Count unknown detections
    unknown_count = conn.execute(
        "SELECT COUNT(*) FROM detections WHERE status = 'Unknown'"
    ).fetchone()[0]

    # Count active alerts
    alert_count = conn.execute(
        "SELECT COUNT(*) FROM alerts WHERE status = 'Active'"
    ).fetchone()[0]

    conn.close()

    return render_template(
        "dashboard.html",
        username=session["username"],
        person_count=person_count,
        vehicle_count=vehicle_count,
        unknown_count=unknown_count,
        alert_count=alert_count
    )


# =========================================================
# KNOWN PERSONS
# =========================================================

@app.route("/known-persons")
def known_persons():

    if "username" not in session:

        return redirect("/")

    conn = get_db_connection()

    persons = conn.execute(
        "SELECT * FROM persons ORDER BY id DESC"
    ).fetchall()

    conn.close()

    return render_template(
        "known_persons.html",
        persons=persons
    )



# ====================================
# ADD KNOWN PERSON
# =========================================================

@app.route("/add-known-person", methods=["POST"])
def add_known_person():

    if "username" not in session:

        return redirect("/")

    name = request.form.get("name")

    person_id = request.form.get("person_id")

    description = request.form.get("description")

    photo = request.files.get("photo")

    photo_filename = None

    if photo and photo.filename:

        filename = secure_filename(photo.filename)

        photo_filename = filename

        photo.save(
            os.path.join(
                app.config["UPLOAD_FOLDER"],
                photo_filename
            )
        )

    conn = get_db_connection()

    conn.execute(
        """
        INSERT INTO persons
        (name, person_id, description, photo)
        VALUES (?, ?, ?, ?)
        """,
        (
            name,
            person_id,
            description,
            photo_filename
        )
    )

    conn.commit()

    conn.close()

    # Retrain face recognition
    train_face_recognizer()

    print("FACE NAMES:", face_names)

    return redirect("/known-persons")


# =========================================================
# SERVE UPLOADED FILES
# =========================================================

@app.route("/uploads/<filename>")
def uploaded_file(filename):

    return send_from_directory(
        app.config["UPLOAD_FOLDER"],
        filename
    )


# =========================================================
# KNOWN VEHICLES
# =========================================================

@app.route("/known-vehicles")
def known_vehicles():

    if "username" not in session:

        return redirect("/")

    conn = get_db_connection()

    vehicles = conn.execute(
        "SELECT * FROM vehicles ORDER BY id DESC"
    ).fetchall()

    conn.close()

    return render_template(
        "known_vehicles.html",
        vehicles=vehicles
    )


# =========================================================
# ADD KNOWN VEHICLE
# =========================================================

@app.route("/add-known-vehicle", methods=["POST"])
def add_known_vehicle():

    if "username" not in session:

        return redirect("/")

    vehicle_number = request.form.get(
        "vehicle_number"
    )

    owner_name = request.form.get(
        "owner_name"
    )

    brand = request.form.get(
        "brand"
    )

    model = request.form.get(
        "model"
    )

    colour = request.form.get(
        "colour"
    )

    conn = get_db_connection()

    conn.execute(
        """
        INSERT INTO vehicles
        (
            vehicle_number,
            owner_name,
            brand,
            model,
            colour
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            vehicle_number,
            owner_name,
            brand,
            model,
            colour
        )
    )

    conn.commit()

    conn.close()

    return redirect("/known-vehicles")


# =========================================================
# SECURITY ALERTS
# =========================================================

@app.route("/security-alerts")
def security_alerts():

    if "username" not in session:

        return redirect("/")

    conn = get_db_connection()

    alerts = conn.execute(
        """
        SELECT *
        FROM alerts
        ORDER BY id DESC
        """
    ).fetchall()

    conn.close()

    return render_template(
        "security_alerts.html",
        alerts=alerts
    )


# =========================================================
# CREATE TEST ALERT
# =========================================================

@app.route("/create-test-alert")
def create_test_alert():

    if "username" not in session:

        return redirect("/")

    conn = get_db_connection()

    # Add test detection

    conn.execute(
        """
        INSERT INTO detections
        (
            object_type,
            name,
            status,
            confidence
        )
        VALUES (?, ?, ?, ?)
        """,
        (
            "Person",
            "Unknown Person",
            "Unknown",
            0
        )
    )

    # Add alert

    conn.execute(
        """
        INSERT INTO alerts
        (
            alert_type,
            message,
            status
        )
        VALUES (?, ?, ?)
        """,
        (
            "Unknown Person",
            "Unknown person detected at camera",
            "Active"
        )
    )

    conn.commit()

    conn.close()

    return redirect("/security-alerts")


# =========================================================
# LIVE TRACKING
# =========================================================

@app.route("/live-tracking")
def live_tracking():

    if "username" not in session:

        return redirect("/")

    camera = request.args.get(
        "camera",
        "Camera-01"
    )

    return render_template(
        "live_tracking.html",
        camera=camera
    )


# =========================================================
# CAMERAS
# =========================================================

@app.route("/cameras")
def cameras():

    if "username" not in session:

        return redirect("/")

    cameras_data = [

        {
            "name": "Camera-01",
            "location": "Main Entrance",
            "status": "Online"
        },

        {
            "name": "Camera-02",
            "location": "Parking Area",
            "status": "Offline"
        }

    ]

    return render_template(
        "cameras.html",
        cameras=cameras_data
    )


# =========================================================
# FACE DETECTION + RECOGNITION
# =========================================================

@app.route("/detect-frame", methods=["POST"])
def detect_frame():

    if "username" not in session:

        return jsonify({
            "success": False,
            "message": "Not logged in"
        })

    try:

        data = request.json["image"]

        # -------------------------------------------------
        # REMOVE BASE64 HEADER
        # -------------------------------------------------

        image_data = data.split(",")[1]

        # -------------------------------------------------
        # BASE64 -> BYTES
        # -------------------------------------------------

        image_bytes = base64.b64decode(
            image_data
        )

        # -------------------------------------------------
        # BYTES -> NUMPY ARRAY
        # -------------------------------------------------

        image_array = np.frombuffer(
            image_bytes,
            dtype=np.uint8
        )

        # -------------------------------------------------
        # NUMPY ARRAY -> OPENCV IMAGE
        # -------------------------------------------------

        frame = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )

        if frame is None:

            return jsonify({
                "success": False,
                "message": "Could not read image"
            })

        # -------------------------------------------------
        # GRAYSCALE
        # -------------------------------------------------

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        # -------------------------------------------------
        # FACE DETECTOR
        # -------------------------------------------------

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50)
        )

        results = []

        # -------------------------------------------------
        # PROCESS EACH FACE
        # -------------------------------------------------

        for (x, y, w, h) in faces:

            face_image = gray[
                y:y+h,
                x:x+w
            ]

            face_image = cv2.resize(face_image, (200, 200))

            name = "Unknown Person"

            status = "Unknown"

            confidence = 0

            # -------------------------------------------------
            # TRY FACE RECOGNITION
            # -------------------------------------------------

            try:

                if len(face_names) > 0:

                    label, distance = face_recognizer.predict(face_image)
                    
                    print("PREDICTED LABEL:", label)
                    print("DISTANCE:", distance)
                    print("FACE NAMES:", face_names)

                    label = int(label)

                    # LBPH:
                    # LOWER distance = better match

                    confidence = max(
                        0,
                        min(
                            100,
                            round(
                                100 - distance,
                                2
                            )
                        )
                    )

                    if (
                        label in face_names
                        and distance < 100
                    ):

                        name = face_names[label]

                        status = "Known"

                    else:

                        name = "Unknown Person"

                        status = "Unknown"

                else:

                    name = "Unknown Person"

                    status = "Unknown"

            except Exception:

                name = "Unknown Person"

                status = "Unknown"

                confidence = 0

            # -------------------------------------------------
            # SAVE DETECTION
            # -------------------------------------------------

            conn = get_db_connection()

            conn.execute(
                """
                INSERT INTO detections
                (
                object_type,
                status,
                confidence
                )
                VALUES (?, ?, ?)
                """,
                (
                    "Face",
                    status,
                    confidence
                )
            )
            

            conn.commit()

            conn.close()

            # -------------------------------------------------
            # UNKNOWN PERSON ALERT
            # -------------------------------------------------

            if status == "Unknown":

                conn = get_db_connection()

                conn.execute(
                    """
                    INSERT INTO alerts
                    (
                        alert_type,
                        message,
                        status
                    )
                    VALUES (?, ?, ?)
                    """,
                    (
                        "Unknown Person",
                        "Unknown person detected by live camera",
                        "Active"
                    )
                )

                conn.commit()

                conn.close()

            # -------------------------------------------------
            # TRACKING RECORD
            # -------------------------------------------------

            conn = get_db_connection()

            conn.execute(
                """
                INSERT INTO tracking
                (
                    name,
                    status,
                    camera
                )
                VALUES (?, ?, ?)
                """,
                (
                    name,
                    status,
                    "Camera-01"
                )
            )

            conn.commit()

            conn.close()

            # -------------------------------------------------
            # RESULT
            # -------------------------------------------------

            results.append({

                "name": name,

                "status": status,

                "confidence": round(
                    float(confidence),
                    2
                )

            })

        # =====================================================
        # RETURN RESULT
        # =====================================================

        return jsonify({

            "success": True,

            "faces": len(faces),

            "results": results

        })

    except Exception as e:

        return jsonify({

            "success": False,

            "message": str(e)

        })


# =========================================================
# LOGOUT
# =========================================================

@app.route("/logout")
def logout():

    session.clear()

    return redirect("/")


# =========================================================
# START APPLICATION
# =========================================================

if __name__ == "__main__":

    init_db()

    # Train known faces when application starts
    train_face_recognizer()

    app.run(
        debug=True
    )