from django.shortcuts import render
from django.http import HttpResponse
#import following packages which will be used to store the file uploaded
from django.conf import settings
from django.core.files.storage import FileSystemStorage


# Create your views here.

def index(request):
    return render(request,'index.html')

def upload_image(request):
    print(request.FILES['image_file'])
    file = request.FILES['image_file']

    fs = FileSystemStorage()
    filename = fs.save(file.name, file)
    uploaded_file_url = fs.url(filename)

    #predict the class of the input image
    img_class, confidence = predict_image(filename)

    return render(request, 'image_upload.html', context = {'uploaded_file_url':uploaded_file_url, 'img_class':img_class, 'confidence': confidence})
 
def predict_image(img_path):
    return "default", 100.00