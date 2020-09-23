from django.contrib.auth import authenticate, login
from django.http import HttpResponse
from django.shortcuts import render,redirect
from django.contrib import messages
from .models import distgrains
from .models import family,distgrains,mem_reg,Registration
from django.contrib.auth.decorators import login_required

from django.contrib.auth.models import User, auth
import cv2
import psycopg2
import datetime
from datetime import date
from dateutil import  relativedelta
import os

import PIL


# Create your views here.
def index(request):
    return render(request, 'index.html')

def family_member(request,id):
    f=family.objects.get(id=id)
    n=mem_reg.objects.filter(uid=f)
    context={
        'n1':n
    }

    return render(request, 'viewmember.html',context)


def updateView(request):
    if request.method == 'POST':
        fname = request.POST['uid']
        print(fname)
        a1=mem_reg.objects.get(aadhar=fname)
        context={
            'a':a1.id
        }

        return render(request,'updateView.html',context)

    else:
        return render(request, 'update.html')


def deleteView(request):
    if request.method == 'POST':
        fname = request.POST['uid']
        print(fname)
        a1=mem_reg.objects.get(aadhar=fname)
        a1.delete()

        return render(request,'facecaptured.html')

    else:
        return render(request, 'deletev.html')
def saveUpdate(request,id):
    if request.method == 'POST':
        fname = request.POST['first_name']
        lname = request.POST['last_name']
        dob = request.POST['dob']
        mobile = request.POST['mobileno']
        m=mem_reg.objects.get(id=id)
        m.first_name=fname
        m.last_name=lname
        m.dob=dob
        m.mobileno=mobile
        m.save()
        return render(request,'facecaptured.html')




def Newmem_reg(request):

    if request.method == 'POST':
        fname = request.POST['first_name']
        lname = request.POST['last_name']
        dob = request.POST['dob']
        gender = request.POST['gender']
        mobile = request.POST['mobileno']
        aadhar = request.POST['aadhar']
        u_id = request.POST['uid']
        r1=family.objects.get(id=u_id)
        r=mem_reg(first_name=fname,last_name=lname,dob=dob,gender=gender,mobileno=mobile,aadhar=aadhar,uid=r1,a=r1.id)
        r.save()
        print("Inserted successfully........")


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
        print(request.user)
        user=auth.authenticate(username=uname,password=upsw)
        if user is not None:
            auth.login(request,user)
            return redirect('dist')
    else:
        return render(request,'index.html')



@login_required
def dist(request):
    f=family.objects.get(id=2)
    d=distgrains.objects.filter(gid=f).filter(dt__month__gte=9).last()
    print(d.dt)
    print(request.user)
    return render(request,'distributer.html')

def Login2(request):

    if request.method=='POST':
        uname=request.POST['uname']
        upsw=request.POST['psw']
        f1=family.objects.filter(aadhar=uname).filter(mobileno=upsw)
        f=family.objects.get(aadhar=uname)

        print(f.id)
        if f1:
            d=distgrains.objects.filter(idd=f.id)
            context={
                'd1' : d,
                'f1':f.id
            }
            return render(request,'user_login.html',context)

    else:
        return render(request,'/')

def newfamily(request):
    return render(request, 'newfamily.html')

@login_required
def capture(request):
    if request.method=='POST':
        fname=request.POST['first_name']
        lname = request.POST['last_name']
        dob = request.POST['dob']
        gender = request.POST['gender']
        mobile = request.POST['mobileno']
        aadhar = request.POST['aadhar']
        card_color = request.POST['card_color']
        f=family.objects.filter(aadhar=aadhar)
        n=mem_reg.objects.filter(aadhar=aadhar)
        #print(n.id)

        if n or f:
            messages.error(request,'Aadhar number already exists with family id:')
            return render(request, 'newfamily.html')
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
            print(path)

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

            f=family(first_name=fname,last_name=lname,dob=dob,gender=gender,mobileno=mobile,aadhar=aadhar,card_color=card_color)
            f.save()
            r=mem_reg(first_name=fname,last_name=lname,dob=dob,gender=gender,mobileno=mobile,aadhar=aadhar,uid=f,a=f.id)
            r.save()
            return render(request,'distributer.html')

    else:
            return render(request,'erro.html')


def distgrain(request):

    return render(request,'distgrains.html')

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
            if (confidence < 100):
                confidence = "  {0}".format(round(100 - confidence))
                k=int(confidence)
                print(k)
                if(k>=45):
                    cam.release()
                    cv2.destroyAllWindows()
                   
                    n=mem_reg.objects.get(aadhar=face_id)
                    f=family.objects.get(id=n.uid.id)
                    n2=mem_reg.objects.filter(a=n.uid.id).count()
                    import datetime
                    now = datetime.datetime.now()
                    y=now.year-18
                    print(y)
                    n3=mem_reg.objects.filter(aadhar=face_id).filter(dob__year__lte=y)
                    if n3:
                        print('ajay')

                    n1=n2
                    print(n1)

                    if n.uid.card_color == "White" :
                        return render(request, 'distgrains2.html')
                    else:
                        
                        if n3:
                            d=distgrains(dal=n1,wheat=3*n1,rice=2*n1,gid=f,idd=f.id)
                            d.save()
                            context={
                                'd1':d,
                                'f1':f,
                                'n2':n1
                            }
                            return render(request, 'distgrains1.html',context)
                        else:

                            return render(request,'chota.html')
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))

            cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

        cv2.imshow('camera', img)

        k = cv2.waitKey(10) & 0xff  
        if k == 27:
            break
    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()
    return render(request,'error.html')