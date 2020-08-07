from django.db import models

# Create your models here.
class UploadFile(models.Model):
    fid = models.IntegerField(default=0)
    name = models.TextField(default="None")
    title = models.TextField(default="") 
    text = models.TextField(default="None")
    eof = models.TextField(null=True)
    lemmas = models.TextField(null=True)

    class Meta:
        db_table = "Uploaded_file"
    def __int__(self):
        return self.fid

class Index(models.Model):
    lemma = models.TextField(default="None")
    position = models.TextField(null=False)
    mesh = models.TextField(null=True)

    class Meta:
        db_table = "Index"
    def __int__(self):
        return self.lemma

class MeshWord(models.Model):
    lemma = models.TextField(default="None")
    mesh = models.TextField(null=False)
    
    class Meta:
        db_table = "MeshWord"