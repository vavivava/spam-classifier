from django.db import models


class PredResults(models.Model):

    input_the_text = models.CharField(max_length=400)
    classification = models.CharField(max_length=30)

    def __str__(self):
        return self.classification
