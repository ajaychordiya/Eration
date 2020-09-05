from django.contrib.auth import authenticate, login
from django.http import HttpResponse
from django.shortcuts import render,redirect
from django.contrib import messages
from .models import distgrains
from .models import family

from django.contrib.auth.models import User, auth
import cv2
import psycopg2
import datetime
from datetime import date
from dateutil import  relativedelta
import os


# Create your views here.
def index(request):
    return render(request, 'index.html')




def Newmem_reg(request):

    if request.method == 'POST':
        fname = request.POST['first_name']
        lname = request.POST['last_name']
        dob = request.POST['dob']
        gender = request.POST['gender']
        mobile = request.POST['mobileno']
        aadhar = request.POST['aadhar']
        u_id = request.POST['uid']

        import psycopg2

        # Establishing the connection
        conn = psycopg2.connect(
            database="Ration", user='postgres', password='ajay', host='127.0.0.1', port='5432'
        )
        # Creating a cursor object using the cursor() method
        cursor = conn.cursor()
        conn.autocommit = True

        # Doping EMPLOYEE table if already exists.

        cursor.execute(
            'INSERT INTO ration_newmem_reg(first_name,last_name,dob,gender,mobileno,aadhar,uid_id) VALUES (%s,%s,%s,%s,%s,%s,%s)',
            (fname, lname, dob, gender, mobile, aadhar, u_id))
       # cursor.execute("UPDATE ration_family SET member_count=member_count + 1 where id =%s", (u_id,))
        conn.commit()
        print("Inserted successfully........")

        conn.close()

        cam = cv2.VideoCapture(0)
        cam.set(3, 640)  # set video width
        cam.set(4, 480)  # set video height

        # make sure 'haarcascade_frontalface_default.xml' is in the same folder as this code
        face_detector = cv2.CascadeClassifier('capt/haarcascade_frontalface_default.xml')

        # For each person, enter one numeric face id (must enter number start from 1, this is the lable of person 1)
        face_id = aadhar


        # Initialize individual sampling face count
        count = 0

        # start detect your face and take 30 pictures
        while (True):

            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                count += 1

                # Save the captured image into the datasets folder
                cv2.imwrite("capt/dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y + h, x:x + w])

                cv2.imshow('image', img)

            k = cv2.waitKey(100) & 0xff  # Press 'ESC' for exiting video
            if k == 27:
                break
            elif count >= 30:  # Take 30 face sample and stop video
                break

        # Do a bit of cleanup
        print("\n [INFO] Exiting Program and cleanup stuff")
        cam.release()
        cv2.destroyAllWindows()

        import numpy as np
        from PIL import Image  # pillow package
        import os
        # Path for face image database
        path = 'capt/dataset'

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        detector = cv2.CascadeClassifier("capt/haarcascade_frontalface_default.xml");

        # function to get the images and label data
        def getImagesAndLabels(path):

            imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
            faceSamples = []
            ids = []

            for imagePath in imagePaths:

                PIL_img = Image.open(imagePath).convert('L')  # convert it to grayscale
                img_numpy = np.array(PIL_img, 'uint8')

                id = int(os.path.split(imagePath)[-1].split(".")[1])
                faces = detector.detectMultiScale(img_numpy)

                for (x, y, w, h) in faces:
                    faceSamples.append(img_numpy[y:y + h, x:x + w])
                    ids.append(id)

            return faceSamples, ids

        print("\n [INFO] Training faces. It will take a few seconds. Wait ...")
        faces, ids = getImagesAndLabels(path)
        recognizer.train(faces, np.array(ids))

        # Save the model into trainer/trainer.yml
        recognizer.write('capt/trainer/' + str(face_id) + ".yml")  # recognizer.save() worked on Mac, but not on Pi

        # Print the numer of faces trained and end program
        print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))


















        return render(request, 'facecaptured.html')
    else:

        return render(request, 'error.html')

def Newmem_page(request):
    return render(request,'Newmem_reg.html')

def Login(request):

    if request.method=='POST':
        uname=request.POST['uname']
        upsw=request.POST['psw']

        user=auth.authenticate(username=uname,password=upsw)
        if user is not None:
            auth.login(request,user)


            return render(request,'../template/distributer.html')

        #else:
            #messages.info(request,'invalid credentials')
    else:
        return render(request,'index.html')

def Login2(request):

    if request.method=='POST':
        uname=request.POST['uname']
        upsw=request.POST['psw']

        conn = psycopg2.connect(database="Ration", user='postgres', password='ajay', host='127.0.0.1', port='5432')
        # Creating a cursor object using the cursor() method
        cursor = conn.cursor()
        conn.autocommit = True
        cursor.execute('SELECT aadhar FROM ration_family where id=%s',(uname,))
        row=cursor.fetchall()
        for r in row:
            adhr=r[0]
        if (upsw==adhr):

            wheat = 0
            rice = 0
            dal = 0

            # cursor.execute('INSERT INTO ration_disthgrains'
            print("hellloo")
            mcount1 = 0
            now1 = ""
            cc = ""
            cursor.execute("SELECT member_count,card_color FROM ration_family where id=%s", (uname,))
            row = cursor.fetchall()
            for r in row:
                mcount1 = r[0]
                cc=r[1]
            print(mcount1)
            # cursor.execute('SELECT dt FROM ration_distgrains where guid_id=%s', (idddd,))
            # row = cursor.fetchall()
            # for r in row:
            # dt = r[0]
            # print(dt)
            idddd=uname
            mcount=mcount1

            wheat = mcount * 5
            rice = mcount * 4
            dal = mcount * 3
            cursor.execute('SELECT dt FROM ration_distgrains where guid_id=%s',(uname,))
            row=cursor.fetchall()
            for r in row:
                now11=r[0]
            print(now11)
            now1=str(now11)
            conn.commit()
            conn.close()
            if cc == "White":
                return render(request, 'distgrains2.html', {'idddd': idddd, 'mcount': mcount})
            else:
                return render(request, 'distgrains1.html',
                          {'mcount': mcount, 'wheat': wheat, 'rice': rice, 'dal': dal, 'idddd': idddd, 'cc': cc,
                           'now1': now1})

    #else:
            #messages.info(request,'invalid credentials')

    else:
        return render(request,'/')

def newfamily(request):
    return render(request, 'newfamily.html')

def capture(request):
    if request.method=='POST':
        fname=request.POST['first_name']
        lname = request.POST['last_name']
        dob = request.POST['dob']
        gender = request.POST['gender']
        mobile = request.POST['mobileno']
        aadhar = request.POST['aadhar']
        card_color = request.POST['card_color']



        conn=psycopg2.connect(database="Ration",user="postgres",password="ajay",host="127.0.0.1",port="5432")
        cursor=conn.cursor()
        conn.autocommit=True
        cursor.execute("SELECT id,aadhar FROM ration_family where aadhar=%s",(aadhar,))
        row=cursor.fetchall()
        adr=0
        for r in row:
            id2=r[0]
            adr=r[1]
        if aadhar==adr:
            messages.error(request,'Aadhar number already exists with family id:')
            return render(request, 'newfamily.html',{'id2': id2})
        else:
            cam = cv2.VideoCapture(0)
            cam.set(3, 640)  # set video width
            cam.set(4, 480)  # set video height

        # make sure 'haarcascade_frontalface_default.xml' is in the same folder as this code
            face_detector = cv2.CascadeClassifier('capt/haarcascade_frontalface_default.xml')

        # For each person, enter one numeric face id (must enter number start from 1, this is the lable of person 1)
            face_id = aadhar

            print("\n [INFO] Initializing face capture. Look the camera and wait ...")
        # Initialize individual sampling face count
            count = 0

        # start detect your face and take 30 pictures
            while (True):

                ret, img = cam.read()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_detector.detectMultiScale(gray, 1.3, 5)

                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    count += 1

                # Save the captured image into the datasets folder
                    cv2.imwrite("capt/dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y + h, x:x + w])

                    cv2.imshow('image', img)

                k = cv2.waitKey(100) & 0xff  # Press 'ESC' for exiting video
                if k == 27:
                    break
                elif count >= 30:  # Take 30 face sample and stop video
                    break

        # Do a bit of cleanup
            print("\n [INFO] Exiting Program and cleanup stuff")
            cam.release()
            cv2.destroyAllWindows()

            import numpy as np
            from PIL import Image  # pillow package
            import os
        # Path for face image database
            path = 'capt/dataset'

            recognizer = cv2.face.LBPHFaceRecognizer_create()
            detector = cv2.CascadeClassifier("capt/haarcascade_frontalface_default.xml");

        # function to get the images and label data
            def getImagesAndLabels(path):

                imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
                faceSamples = []
                ids = []

                for imagePath in imagePaths:

                    PIL_img = Image.open(imagePath).convert('L')  # convert it to grayscale
                    img_numpy = np.array(PIL_img, 'uint8')

                    id = int(os.path.split(imagePath)[-1].split(".")[1])
                    faces = detector.detectMultiScale(img_numpy)

                    for (x, y, w, h) in faces:
                        faceSamples.append(img_numpy[y:y + h, x:x + w])
                        ids.append(id)

                    return faceSamples, ids

            print("\n [INFO] Training faces. It will take a few seconds. Wait ...")
            faces, ids = getImagesAndLabels(path)
            recognizer.train(faces, np.array(ids))

        # Save the model into trainer/trainer.yml
            recognizer.write('capt/trainer/'+ str(face_id) + ".yml")  # recognizer.save() worked on Mac, but not on Pi

        # Print the numer of faces trained and end program
            print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))


        # Establishing the connection
            conn = psycopg2.connect(
                database="Ration", user='postgres', password='ajay', host='127.0.0.1', port='5432'
                )
        #Creating a cursor object using the cursor() method
            cursor = conn.cursor()
            conn.autocommit = True

        # Doping EMPLOYEE table if already exists.

            cursor.execute('INSERT INTO ration_family(first_name,last_name,dob,gender,mobileno,aadhar,card_color) VALUES (%s,%s,%s,%s,%s,%s,%s)',(fname, lname, dob, gender, mobile, aadhar,card_color))
            conn.commit()
            print("Inserted successfully........")
            if (card_color=="Yellow" or card_color=="Orange"):
                cursor.execute('SELECT id FROM ration_family where aadhar=%s',(aadhar,))
                row=cursor.fetchall()
                for r in row:
                    gid=r[0]
                cursor.execute('INSERT INTO ration_distgrains(dt,guid_id) VALUES (%s,%s)',("0",gid))
                conn.commit()
            conn.close()
            return render(request,'distributer.html')

    else:
            return render(request,'erro.html')


def distgrains(request):

    return render(request,'distgrains.html')

def addgrains(request):

    if request.method == 'POST':
        wheat = request.POST['wheat']
        rice = request.POST['rice']
        dal = request.POST['dal']
        gid = request.POST['uid']

        import psycopg2

        # Establishing the connection
        conn = psycopg2.connect(
            database="Ration", user='postgres', password='ajay', host='127.0.0.1', port='5432'
        )
        # Creating a cursor object using the cursor() method
        cursor = conn.cursor()
        conn.autocommit = True

        # Doping EMPLOYEE table if already exists.

        cursor.execute(
            'INSERT INTO ration_distgrains(wheat,rice,dal,guid_id) VALUES (%s,%s,%s,%s,%s)',
            (wheat, rice, dal, gid))
        conn.commit()
        print("Inserted successfully........")

        conn.close()

        return render(request, 'facecaptured.html')
    else:
        return render(request, 'error.html')





    return render((request,'facecaptured.html'))

def gdata(request):

    gdata1=request.POST['uid']
    #if request.method=='POST':
     #   id3=request.POST['id3']
    # Establishing the connection
    #now = datetime.datetime.now()
    #now2 = now.strftime("%m-%y")
    #conn = psycopg2.connect(
     #   database="Ration", user='postgres', password='808748', host='127.0.0.1', port='5432'
    #)
    # Creating a cursor object using the cursor() method
    #cursor = conn.cursor()
    #conn.autocommit = True
    #cursor.execute("UPDATE ration_distgrains SET dist=1 where guid_id=%s",(id3,))
    #conn.commit()

    #cursor.execute(
     #   'INSERT INTO ration_distgrains(guid_id,dt) VALUES (%s,%s)',
      #  (idddd, now2))
    #conn.commit()
    #print("Inserted successfully........")
    #print("Updated successfully........")

    # Doping EMPLOYEE table if already exists.
    #conn.close()

def recognize(request):
    import cv2
    import numpy as np
    import os

    face_id = request.POST['uid']
    face_id1="capt/trainer/"+str(face_id)+".yml"

    print(face_id1)



    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(face_id1)  # load trained model
    cascadePath = "capt/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath);

    font = cv2.FONT_HERSHEY_SIMPLEX
    # Initialize and start realtime video capture
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)  # set video widht
    cam.set(4, 480)  # set video height

    # Define min window size to be recognized as a face
    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)

    while True:

        ret, img = cam.read()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW), int(minH)),
        )
        for (x, y, w, h) in faces:

            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

            # Check if confidence is less them 100 ==> "0" is perfect match
            if (confidence < 100):
                confidence = "  {0}".format(round(100 - confidence))
                k=int(confidence)
                print(k)
                if(k>=40):
                    cam.release()
                    cv2.destroyAllWindows()
                    conn = psycopg2.connect(database="Ration", user='postgres', password='ajay', host='127.0.0.1',
                                            port='5432')
                    # Creating a cursor object using the cursor() method
                    cursor = conn.cursor()
                    conn.autocommit = True
















                    cursor.execute("SELECT * FROM ration_newmem_reg where aadhar =%s",(face_id,))
                    #conn.commit()
                    row = cursor.fetchall()
                    print(row)
                    for r in row:
                        id = str(r[7])

                    #cursor.execute("SELECT *FROM ration_distgrains where guid_id =%s", (id,))
                    conn.commit()

                    row = cursor.fetchall()
                    print(row)
                    cursor.execute("SELECT uid_id FROM ration_newmem_reg where aadhar =%s",(face_id,))
                    row=cursor.fetchall()
                    print("helllo")

                    print(row)
                    iddd=0
                    mcount=0
                    idddd=0

                    cc=""
                    for r in row:
                        iddd=r[0]
                    print(iddd)
                    if iddd:
                        cursor.execute("SELECT id FROM ration_family where id=%s",(iddd,))
                        row=cursor.fetchall()
                        print(row)
                        for r in row:
                            mcount = 3
                            idddd = r[0]
                        print(mcount)
                    else:
                        cursor.execute("SELECT id FROM ration_family where aadhar=%s",(face_id,))
                        row=cursor.fetchall()
                        print(row)
                        for r in row:
                            mcount = 3
                            idddd = r[0]
                        print(mcount)
                    #cursor.execute('SELECT dt FROM ration_distgrains where guid_id=%s', (idddd,))
                    #row = cursor.fetchall()
                    #for r in row:
                        #dt = r[0]
                    #print(dt)

                    cursor.execute("SELECT card_color FROM ration_family where id=%s", (idddd,))
                    row = cursor.fetchall()
                    for r in row:
                        cc = r[0]
                    print(cc)
                    if cc == "White":
                        return render(request, 'distgrains2.html', {'idddd': idddd, 'mcount': mcount})
                    else:
                        cursor.execute('SELECT EXTRACT(YEAR FROM dt) FROM ration_distgrains')
                        yy = cursor.fetchall()



                        #print("year", yy[1])

                        now = datetime.datetime.now()
                        now1 = now.strftime("%d-%m-%y     %H:%M")
                        now2 = now.strftime("%m-%d-%y")
                        now3=now2




                        wheat = 0
                        rice = 0
                        dal = 0
                        wheat = mcount * 5
                        rice = mcount * 4
                        dal = mcount * 3

                        cursor.execute('INSERT INTO ration_distgrains(guid_id,wheat,rice,dal,dt,id3) VALUES (%s,%s,%s,%s,%s,%s)',(idddd,wheat,rice,dal,now2,'2'))
                        conn.commit()
                        conn.close()
                        return render(request, 'distgrains1.html',{'mcount': mcount, 'wheat': wheat, 'rice': rice, 'dal': dal, 'idddd': idddd,'cc': cc, 'now1': now1})

                    #cursor.execute("SELECT card_color FROM ration_family where id=%s",(idddd,))
                    #row=cursor.fetchall()
                    #for r in row:
                     #   cc=r[0]
                    #print(cc)


                    #if cc=="White":
                     #   return render(request,'distgrains2.html',{'idddd':idddd,'mcount':mcount})
                    #else:
                        #cursor.execute('INSERT INTO ration_distgrains(dt,guid_id) VALUES(%s,%s)',(now2,idddd))
                        #conn.commit()
                        #conn.close()
                        #return render(request,'distgrains1.html',{'mcount':mcount,'wheat':wheat,'rice':rice,'dal':dal,'idddd':idddd,'cc':cc,'now1':now1})

            #else:
                    #messages.error(request, 'Face do not match')
                    #return render(request,'distgrains.html')
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))

            cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

        cv2.imshow('camera', img)

        k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
        if k == 27:
            break

    # Do a bit of cleanup
    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()
    return render(request,'error.html')