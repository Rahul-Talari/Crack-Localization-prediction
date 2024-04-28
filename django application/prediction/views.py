

from django.http import JsonResponse
from rest_framework.decorators import api_view

@api_view(['GET'])
def add(request):
    num1 = int(request.GET['num1'])
    num2 = int(request.GET['num2'])
    result = num1 + num2
    return JsonResponse({'result': result})
