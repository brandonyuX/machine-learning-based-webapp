from django.shortcuts import render
from django.http import HttpResponse
#import following packages which will be used to store the file uploaded
from django.conf import settings
from django.core.files.storage import FileSystemStorage

from .models import UploadedImage
from .forms import UploadImageForm
 
from torchvision import models
import torch
from torchvision import transforms
from PIL import Image #to load the uploaded image
# Create your views here.
alexnet = models.alexnet(pretrained = True)


def index(request):
    file_path = None
    if request.method == 'POST':
        form = UploadImageForm(request.POST, request.FILES)
        if form.is_valid():
            instance = form.save()
            # Obtain the file path of the saved image
            file_path = instance.image.path
            last_uploaded_image = UploadedImage.objects.last()
            img_class, confidence = predict_image(file_path)
            return render(request, 'index.html', {'form': form, 'file_path': file_path, 'last_uploaded_image': last_uploaded_image,'img_class':img_class, 'confidence': confidence})
    else:
        form = UploadImageForm()

    last_uploaded_image = UploadedImage.objects.last()
    return render(request, 'index.html', {'form': form, 'file_path': file_path, 'last_uploaded_image': last_uploaded_image})


def predict_image(img_path):
    #define the transformations that needs to be apply before passing the image into the model 
    transform = transforms.Compose([            #[1]
                transforms.Resize(256),                    #[2]
                transforms.CenterCrop(224),                #[3]
                transforms.ToTensor(),                     #[4]
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])                  #[7]
                ])

    #below command will load the image and convert it into 3 channel image as the model requires 3 channel image
    #img = Image.open(settings.MEDIA_ROOT+'/'+img_path).convert('RGB')
    img = Image.open(img_path).convert('RGB')
    #transformm the image by applying the transformations defined above
    img_t = transform(img)

    #adding one more dimension for batches as model takes the input in batches of images but we have only one image here so we will have one batch containing 1 image
    batch_t = torch.unsqueeze(img_t, 0)

    #put the model in eval mode
    alexnet.eval()

    #getting the output from model
    out = alexnet(batch_t)

    #load the image net classes you need to put the imagenet classes text file in you computer and use its location here
    with open("C:/Users/binba/OneDrive/Desktop/Project/machine-learning-based-webapp/image_classifier/media/imagenet-classes.txt") as f:
        classes = [line.strip() for line in f.readlines()]

    _, index = torch.max(out, 1)
    
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
    print(classes[index[0]],percentage[index[0]].item())
    return (classes[index[0]], percentage[index[0]].item())