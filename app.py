from flask import Flask, render_template, request, session, redirect, url_for, Response, jsonify
import mysql.connector
import cv2
from PIL import Image
import numpy as np
import os
import time
from datetime import date

app = Flask(__name__)

cnt = 0
pause_cnt = 0
justscanned = False

mydb = mysql.connector.connect(
    host=" sql8.freesqldatabase.com",
    user="sql8658807",
    passwd="mJHsujLEjA",
    database="sql8658807"
)
mycursor = mydb.cursor()


# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Generate dataset >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def generate_dataset(nbr):
    face_classifier = cv2.CascadeClassifier(
        # "C:/Users/kelvi/PycharmProjects/FlaskOpencv_FaceRecognition/resources/haarcascade_frontalface_default.xml")
        "resources/haarcascade_frontalface_default.xml"        
        )
    
    def face_cropped(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        # scaling factor=1.3
        # Minimum neighbor = 5

        if faces is ():
            return None
        for (x, y, w, h) in faces:
            cropped_face = img[y:y + h, x:x + w]
        return cropped_face

    cap = cv2.VideoCapture(0)

    mycursor.execute("select ifnull(max(img_id), 0) from img_dataset")
    row = mycursor.fetchone()
    lastid = row[0]

    img_id = lastid
    max_imgid = img_id + 100
    count_img = 0

    while True:
        ret, img = cap.read()
        if face_cropped(img) is not None:
            count_img += 1
            img_id += 1
            face = cv2.resize(face_cropped(img), (200, 200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            file_name_path = "dataset/" + nbr + "." + str(img_id) + ".jpg"
            cv2.imwrite(file_name_path, face)
            cv2.putText(face, str(count_img), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

            mycursor.execute("""INSERT INTO `img_dataset` (`img_id`, `img_person`) VALUES
                                ('{}', '{}')""".format(img_id, nbr))
            mydb.commit()

            frame = cv2.imencode('.jpg', face)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            if cv2.waitKey(1) == 13 or int(img_id) == int(max_imgid):
                break
                cap.release()
                cv2.destroyAllWindows()


# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Train Classifier >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
@app.route('/train_classifier/<nbr>')
def train_classifier(nbr):
    # dataset_dir = "C:/Users/kelvi/PycharmProjects/FlaskOpencv_FaceRecognition/dataset"
    dataset_dir = "dataset"

    path = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir)]
    faces = []
    ids = []

    for image in path:
        img = Image.open(image).convert('L');
        imageNp = np.array(img, 'uint8')
        id = int(os.path.split(image)[1].split(".")[1])

        faces.append(imageNp)
        ids.append(id)
    ids = np.array(ids)

    # Train the classifier and save
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)
    clf.write("classifier.xml")

    return redirect('/')


# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Face Recognition >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def face_recognition():  # generate frame by frame from camera
    def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf):
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features = classifier.detectMultiScale(gray_image, scaleFactor, minNeighbors)

        global justscanned
        global pause_cnt

        pause_cnt += 1

        coords = []

        for (x, y, w, h) in features:
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            id, pred = clf.predict(gray_image[y:y + h, x:x + w])
            confidence = int(100 * (1 - pred / 300))

            if confidence > 70 and not justscanned:
                global cnt
                cnt += 1

                n = (100 / 30) * cnt
                # w_filled = (n / 100) * w
                w_filled = (cnt / 30) * w

                cv2.putText(img, str(int(n)) + ' %', (x + 20, y + h + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (153, 255, 255), 2, cv2.LINE_AA)

                cv2.rectangle(img, (x, y + h + 40), (x + w, y + h + 50), color, 2)
                cv2.rectangle(img, (x, y + h + 40), (x + int(w_filled), y + h + 50), (153, 255, 255), cv2.FILLED)

                mycursor.execute("select a.img_person, b.prs_name, b.prs_skill "
                                 "  from img_dataset a "
                                 "  left join prs_mstr b on a.img_person = b.prs_nbr "
                                 " where img_id = " + str(id))
                row = mycursor.fetchone()
                pnbr = row[0]
                pname = row[1]
                pskill = row[2]

                if int(cnt) == 30:
                    cnt = 0

                    mycursor.execute("insert into accs_hist (accs_date, accs_prsn) values('" + str(
                        date.today()) + "', '" + pnbr + "')")
                    mydb.commit()

                    cv2.putText(img, pname + ' | ' + pskill, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (153, 255, 255), 2, cv2.LINE_AA)
                    time.sleep(1)

                    justscanned = True
                    pause_cnt = 0

            else:
                if not justscanned:
                    cv2.putText(img, 'UNKNOWN', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    cv2.putText(img, ' ', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

                if pause_cnt > 80:
                    justscanned = False

            coords = [x, y, w, h]
        return coords

    def recognize(img, clf, faceCascade):
        coords = draw_boundary(img, faceCascade, 1.1, 10, (255, 255, 0), "Face", clf)
        return img

    faceCascade = cv2.CascadeClassifier(
        "resources/haarcascade_frontalface_default.xml"
        )
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.read("classifier.xml")

    wCam, hCam = 400, 400

    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)

    while True:
        ret, img = cap.read()
        img = recognize(img, clf, faceCascade)

        frame = cv2.imencode('.jpg', img)[1].tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

        key = cv2.waitKey(1)
        if key == 27:
            break


@app.route('/')
def home():
    mycursor.execute("select prs_nbr, prs_name, prs_skill, prs_active, prs_added from prs_mstr")
    data = mycursor.fetchall()

    return render_template('index.html', data=data)


@app.route('/addprsn')
def addprsn():
    mycursor.execute("select ifnull(max(prs_nbr) + 1, 101) from prs_mstr")
    row = mycursor.fetchone()
    nbr = row[0]
    # print(int(nbr))

    return render_template('addprsn.html', newnbr=int(nbr))


@app.route('/addprsn_submit', methods=['POST'])
def addprsn_submit():
    prsnbr = request.form.get('txtnbr')
    prsname = request.form.get('txtname')
    prsskill = request.form.get('optskill')

    mycursor.execute("""INSERT INTO `prs_mstr` (`prs_nbr`, `prs_name`, `prs_skill`) VALUES
                    ('{}', '{}', '{}')""".format(prsnbr, prsname, prsskill))
    mydb.commit()

    # return redirect(url_for('home'))
    return redirect(url_for('vfdataset_page', prs=prsnbr))


@app.route('/vfdataset_page/<prs>')
def vfdataset_page(prs):
    return render_template('gendataset.html', prs=prs)


@app.route('/vidfeed_dataset/<nbr>')
def vidfeed_dataset(nbr):
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(generate_dataset(nbr), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed')
def video_feed():
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(face_recognition(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/fr_page')
def fr_page():
    """Video streaming home page."""
    mycursor.execute("select a.accs_id, a.accs_prsn, b.prs_name, b.prs_skill, a.accs_added "
                     "  from accs_hist a "
                     "  left join prs_mstr b on a.accs_prsn = b.prs_nbr "
                     " where a.accs_date = curdate() "
                     " order by 1 desc")
    data = mycursor.fetchall()

    return render_template('fr_page.html', data=data)


@app.route('/countTodayScan')
def countTodayScan():
    mydb = mysql.connector.connect(
        host=" sql8.freesqldatabase.com",
        user="sql8658807",
        passwd="mJHsujLEjA",
        database="sql8658807"
    )
    mycursor = mydb.cursor()

    mycursor.execute("select count(*) "
                     "  from accs_hist "
                     " where accs_date = curdate() ")
    row = mycursor.fetchone()
    rowcount = row[0]

    return jsonify({'rowcount': rowcount})


@app.route('/loadData', methods=['GET', 'POST'])
def loadData():
    mydb = mysql.connector.connect(
        host=" sql8.freesqldatabase.com",
        user="sql8658807",
        passwd="mJHsujLEjA",
        database="sql8658807"
    )
    mycursor = mydb.cursor()

    mycursor.execute("select a.accs_id, a.accs_prsn, b.prs_name, b.prs_skill, date_format(a.accs_added, '%H:%i:%s') "
                     "  from accs_hist a "
                     "  left join prs_mstr b on a.accs_prsn = b.prs_nbr "
                     " where a.accs_date = curdate() "
                     " order by 1 desc")
    data = mycursor.fetchall()

    return jsonify(response=data)


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=True)





#MODIFIED WITH SQLITE

# from flask import Flask, render_template, request, session, redirect, url_for, Response, jsonify
# import cv2
# from PIL import Image
# import numpy as np
# import os
# import time
# from datetime import date
# import sqlite3

# app = Flask(__name__)

# # Create and connect to the SQLite database
# conn = sqlite3.connect('flask_db.sqlite')
# cursor = conn.cursor()

# # Create tables if they don't exist
# cursor.execute('''
#     CREATE TABLE IF NOT EXISTS img_dataset (
#         img_id INTEGER PRIMARY KEY AUTOINCREMENT,
#         img_person TEXT
#     )
# ''')

# cursor.execute('''
#     CREATE TABLE IF NOT EXISTS prs_mstr (
#         prs_nbr TEXT PRIMARY KEY,
#         prs_name TEXT,
#         prs_skill TEXT
#     )
# ''')

# cursor.execute('''
#     CREATE TABLE IF NOT EXISTS accs_hist (
#         accs_id INTEGER PRIMARY KEY AUTOINCREMENT,
#         accs_date DATE,
#         accs_prsn TEXT
#     )
# ''')

# conn.commit()

# # Initialize the SQLite connection for the application
# conn_app = sqlite3.connect('flask_db.sqlite')
# cursor_app = conn_app.cursor()


# # Generate dataset function
# def generate_dataset(nbr):
#     face_classifier = cv2.CascadeClassifier("resources/haarcascade_frontalface_default.xml")

#     def face_cropped(img):
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         faces = face_classifier.detectMultiScale(gray, 1.3, 5)

#         if len(faces) == 0:
#             return None

#         for (x, y, w, h) in faces:
#             cropped_face = img[y:y + h, x:x + w]

#         return cropped_face

#     cap = cv2.VideoCapture(0)

#     cursor_app.execute("SELECT IFNULL(MAX(img_id), 0) FROM img_dataset")
#     row = cursor_app.fetchone()
#     lastid = row[0]

#     img_id = lastid
#     max_imgid = img_id + 100
#     count_img = 0

#     while True:
#         ret, img = cap.read()
#         if face_cropped(img) is not None:
#             count_img += 1
#             img_id += 1
#             face = cv2.resize(face_cropped(img), (200, 200))
#             face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

#             file_name_path = "dataset/{}.{}.jpg".format(nbr, img_id)
#             cv2.imwrite(file_name_path, face)
#             cv2.putText(face, str(count_img), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

#             cursor_app.execute("INSERT INTO img_dataset (img_person) VALUES (?)", (nbr,))
#             conn_app.commit()

#             frame = cv2.imencode('.jpg', face)[1].tobytes()
#             yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

#             if cv2.waitKey(1) == 13 or int(img_id) == int(max_imgid):
#                 cap.release()
#                 cv2.destroyAllWindows()
#                 break


# # Train Classifier function
# @app.route('/train_classifier/<nbr>')
# def train_classifier(nbr):
#     dataset_dir = "dataset"
#     path = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir)]
#     faces = []
#     ids = []

#     for image in path:
#         img = Image.open(image).convert('L')
#         imageNp = np.array(img, 'uint8')
#         id = int(os.path.split(image)[1].split(".")[1])

#         faces.append(imageNp)
#         ids.append(id)

#     ids = np.array(ids)

#     # Train the classifier and save
#     clf = cv2.face.LBPHFaceRecognizer_create()
#     clf.train(faces, ids)
#     clf.write("classifier.xml")

#     return redirect('/')


# # Face Recognition function
# def face_recognition():
#     def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf):
#         gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         features = classifier.detectMultiScale(gray_image, scaleFactor, minNeighbors)

#         global justscanned
#         global pause_cnt

#         pause_cnt += 1
#         coords = []

#         for (x, y, w, h) in features:
#             cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
#             id, pred = clf.predict(gray_image[y:y + h, x:x + w])
#             confidence = int(100 * (1 - pred / 300))

#             if confidence > 70 and not justscanned:
#                 global cnt
#                 cnt += 1

#                 n = (100 / 30) * cnt
#                 w_filled = (cnt / 30) * w

#                 cv2.putText(img, str(int(n)) + ' %', (x + 20, y + h + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
#                             (153, 255, 255), 2, cv2.LINE_AA)

#                 cv2.rectangle(img, (x, y + h + 40), (x + w, y + h + 50), color, 2)
#                 cv2.rectangle(img, (x, y + h + 40), (x + int(w_filled), y + h + 50), (153, 255, 255), cv2.FILLED)

#                 cursor_app.execute("SELECT a.img_person, b.prs_name, b.prs_skill "
#                                    "FROM img_dataset a "
#                                    "LEFT JOIN prs_mstr b ON a.img_person = b.prs_nbr "
#                                    "WHERE img_id = ?", (id,))
#                 row = cursor_app.fetchone()
#                 pnbr = row[0]
#                 pname = row[1]
#                 pskill = row[2]

#                 if int(cnt) == 30:
#                     cnt = 0

#                     cursor_app.execute("INSERT INTO accs_hist (accs_date, accs_prsn) VALUES (?, ?)",
#                                        (str(date.today()), pnbr))
#                     conn_app.commit()

#                     cv2.putText(img, pname + ' | ' + pskill, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
#                                 (153, 255, 255), 2, cv2.LINE_AA)
#                     time.sleep(1)

#                     justscanned = True
#                     pause_cnt = 0
#             else:
#                 if not justscanned:
#                     cv2.putText(img, 'UNKNOWN', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
#                 else:
#                     cv2.putText(img, ' ', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

#                 if pause_cnt > 80:
#                     justscanned = False

#             coords = [x, y, w, h]
#         return
# # ... (Previous code, including the "face_recognition" function)

# @app.route('/')
# def home():
#     cursor_app.execute("SELECT prs_nbr, prs_name, prs_skill, prs_active, prs_added FROM prs_mstr")
#     data = cursor_app.fetchall()

#     return render_template('index.html', data=data)


# @app.route('/addprsn')
# def addprsn():
#     cursor_app.execute("SELECT IFNULL(MAX(prs_nbr) + 1, 101) FROM prs_mstr")
#     row = cursor_app.fetchone()
#     nbr = row[0]

#     return render_template('addprsn.html', newnbr=int(nbr))


# @app.route('/addprsn_submit', methods=['POST'])
# def addprsn_submit():
#     prsnbr = request.form.get('txtnbr')
#     prsname = request.form.get('txtname')
#     prsskill = request.form.get('optskill')

#     cursor_app.execute("INSERT INTO prs_mstr (prs_nbr, prs_name, prs_skill) VALUES (?, ?, ?)",
#                        (prsnbr, prsname, prsskill))
#     conn_app.commit()

#     return redirect(url_for('vfdataset_page', prs=prsnbr))


# @app.route('/vfdataset_page/<prs>')
# def vfdataset_page(prs):
#     return render_template('gendataset.html', prs=prs)


# @app.route('/vidfeed_dataset/<nbr>')
# def vidfeed_dataset(nbr):
#     # Video streaming route. Put this in the src attribute of an img tag
#     return Response(generate_dataset(nbr), mimetype='multipart/x-mixed-replace; boundary=frame')


# @app.route('/video_feed')
# def video_feed():
#     # Video streaming route. Put this in the src attribute of an img tag
#     return Response(face_recognition(), mimetype='multipart/x-mixed-replace; boundary=frame')


# @app.route('/fr_page')
# def fr_page():
#     cursor_app.execute("SELECT a.accs_id, a.accs_prsn, b.prs_name, b.prs_skill, a.accs_added "
#                        "FROM accs_hist a "
#                        "LEFT JOIN prs_mstr b ON a.accs_prsn = b.prs_nbr "
#                        "WHERE a.accs_date = date('now') "
#                        "ORDER BY a.accs_id DESC")
#     data = cursor_app.fetchall()

#     return render_template('fr_page.html', data=data)


# @app.route('/countTodayScan')
# def countTodayScan():
#     cursor_app.execute("SELECT COUNT(*) "
#                        "FROM accs_hist "
#                        "WHERE accs_date = date('now')")
#     row = cursor_app.fetchone()
#     rowcount = row[0]

#     return jsonify({'rowcount': rowcount})


# @app.route('/loadData', methods=['GET', 'POST'])
# def loadData():
#     cursor_app.execute("SELECT a.accs_id, a.accs_prsn, b.prs_name, b.prs_skill, "
#                        "strftime('%H:%M:%S', a.accs_added) "
#                        "FROM accs_hist a "
#                        "LEFT JOIN prs_mstr b ON a.accs_prsn = b.prs_nbr "
#                        "WHERE a.accs_date = date('now') "
#                        "ORDER BY a.accs_id DESC")
#     data = cursor_app.fetchall()

#     return jsonify(response=data)


# if __name__ == "__main__":
#     app.run(host='127.0.0.1', port=5000, debug=True)
