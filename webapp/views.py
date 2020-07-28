from django.shortcuts import render
from django.http import HttpResponse
from django.shortcuts import get_object_or_404
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from . models import employee
from . import serializers
from . import scriptx
import base64
# Create your views here.

class employeeList(APIView):

    serailizer_class = serializers.scriptserializer
    def get(self, request):
        """Humble get man"""
        return Response({"message":"This is get and it is working!"})

    def post(self, request):
        """Post API, will call an external py script and return data"""
        serializer = serializers.scriptserializer(data = request.data)

        if serializer.is_valid():
            goal = request.data.get('goal')
            years = request.data.get('years')
            principal = scriptx.functiony(goal,years)
            with open('testa.png', 'rb') as imagefile:
                base64string = base64.b64encode(imagefile.read()).decode('ascii')
            return Response({'principal':principal, 'chart':base64string})
        else:
            return Response(serailizer.errors, status = status.HTTP_400_BAD_REQUEST)
