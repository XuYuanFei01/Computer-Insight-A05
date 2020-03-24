# This is an auto-generated Django model module.
# You'll have to do the following manually to clean this up:
#   * Rearrange models' order
#   * Make sure each model has one field with primary_key=True
#   * Make sure each ForeignKey and OneToOneField has `on_delete` set to the desired behavior
#   * Remove `managed = False` lines if you wish to allow Django to create, modify, and delete the table
# Feel free to rename the models, but don't rename db_table values or field names.
from django.db import models


class Admin(models.Model):
    name = models.CharField(max_length=30, blank=True, null=True)
    username = models.CharField(max_length=30, blank=True, null=True)
    password = models.CharField(max_length=30, blank=True, null=True)
    photo = models.TextField(blank=True, null=True)
    gender = models.BooleanField(blank=True)
    email = models.TextField(max_length=254,null=True)
    text = models.TextField(null=True)
    class Meta:
        managed = False
        db_table = 'admin'


class Records(models.Model):
    time = models.DateTimeField()
    wear_hat_confidence = models.DecimalField(max_digits=4, decimal_places=4, blank=True, null=True)
    most_close_worker_id = models.CharField(max_length=20,blank=True, null=True)
    name = models.CharField(max_length=30,blank=True, null=True)
    photo = models.TextField(blank=True, null=True)
    isread = models.BooleanField(null=True)

    class Meta:
        managed = False
        db_table = 'records'


class Workers(models.Model):
    workerid = models.CharField(max_length=30)
    name = models.CharField(max_length=10)
    gender = models.TextField(blank=True, null=True)  # This field type is a guess.
    photo = models.TextField(blank=True, null=True)
    face_feature = models.TextField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'workers'



