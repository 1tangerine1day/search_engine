from django.contrib import admin

# Register your models here.
from polls.models import UploadFile
from polls.models import Index
from polls.models import MeshWord

# Register your models here.

class meshWord_colm(admin.ModelAdmin):
    list_display = ('lemma','mesh')
admin.site.register(MeshWord, meshWord_colm)

class index_colm(admin.ModelAdmin):
    list_display = ('lemma','position')
admin.site.register(Index,index_colm)

class uploadFile_colm(admin.ModelAdmin):
    list_display = ('fid','name')
admin.site.register(UploadFile,uploadFile_colm)
