from django import forms
from .models import UploadedImage


class ImageUploadForm(forms.ModelForm):
    """فرم آپلود تصویر"""
    
    class Meta:
        model = UploadedImage
        fields = ['image']
        widgets = {
            'image': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': 'image/*',
                'id': 'imageInput'
            })
        }
        labels = {
            'image': 'انتخاب تصویر'
        }
