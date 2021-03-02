from django.shortcuts import render
from django.http import JsonResponse
from src.message_predict import MessagePredict
from . models import PredResults


def predict(request):
    return render(request, 'predict.html')

def predict_chances(request):

    if request.POST.get('action') == 'post':

        input_the_text = request.POST.get('input_the_text')

        mp = MessagePredict()
        result = mp.message_predict(input_the_text, False)

        classification = float(result[0])

        PredResults.objects.create(input_the_text=input_the_text, classification=classification)

        print('result is', + classification)
        return JsonResponse({'result': classification, 'input_the_text': input_the_text}, safe=False)

def view_results(request):
    data = {"dataset": PredResults.objects.all()}
    return render(request, "results.html", data)
