from django.contrib import admin
from .models import UploadedImage


@admin.register(UploadedImage)
class UploadedImageAdmin(admin.ModelAdmin):
    """پنل مدیریت تصاویر آپلود شده"""
    
    list_display = ['id', 'image_thumbnail', 'has_vehicle', 'vehicle_count', 'uploaded_at']
    list_filter = ['has_vehicle', 'uploaded_at']
    search_fields = ['id']
    readonly_fields = ['image_preview', 'processed_image_preview', 'uploaded_at', 'detection_result']
    
    fieldsets = (
        ('اطلاعات تصویر', {
            'fields': ('image', 'image_preview')
        }),
        ('نتایج تشخیص', {
            'fields': ('has_vehicle', 'vehicle_count', 'detection_result')
        }),
        ('تصویر پردازش شده', {
            'fields': ('processed_image', 'processed_image_preview')
        }),
        ('اطلاعات زمانی', {
            'fields': ('uploaded_at',)
        }),
    )
    
    def image_thumbnail(self, obj):
        """نمایش تصویر کوچک در لیست"""
        if obj.image:
            return f'<img src="{obj.image.url}" style="width: 80px; height: 60px; object-fit: cover; border-radius: 5px;">'
        return '-'
    image_thumbnail.short_description = 'تصویر'
    image_thumbnail.allow_tags = True
    
    def image_preview(self, obj):
        """نمایش تصویر اصلی در صفحه جزئیات"""
        if obj.image:
            return f'<img src="{obj.image.url}" style="max-width: 500px; border-radius: 10px;">'
        return '-'
    image_preview.short_description = 'پیش‌نمایش تصویر اصلی'
    image_preview.allow_tags = True
    
    def processed_image_preview(self, obj):
        """نمایش تصویر پردازش شده در صفحه جزئیات"""
        if obj.processed_image:
            return f'<img src="{obj.processed_image.url}" style="max-width: 500px; border-radius: 10px;">'
        return 'تصویر پردازش شده وجود ندارد'
    processed_image_preview.short_description = 'پیش‌نمایش تصویر پردازش شده'
    processed_image_preview.allow_tags = True
