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
            date = request.data.get('date')
            principal,avg_savings,profit,purchase_possible,fd_principal,score = scriptx.functiony(goal,date)
            with open('spending-by-category.png', 'rb') as imagefile:
                chart1 = base64.b64encode(imagefile.read()).decode('ascii')
                chart1 = "data:image/png;base64,"+chart1
            with open('monthly-total-transactions.png', 'rb') as imagefile:
                chart2 = base64.b64encode(imagefile.read()).decode('ascii')
                chart2 = "data:image/png;base64,"+chart2
            with open('monthly_debit_credit.png', 'rb') as imagefile:
                chart3 = base64.b64encode(imagefile.read()).decode('ascii')
                chart3 = "data:image/png;base64,"+chart3
            with open('monthy-savings.png', 'rb') as imagefile:
                chart4 = base64.b64encode(imagefile.read()).decode('ascii')
                chart4 = "data:image/png;base64,"+chart4
            with open('savings-vs-rd.png', 'rb') as imagefile:
                chart5 = base64.b64encode(imagefile.read()).decode('ascii')
                chart5 = "data:image/png;base64,"+chart5
            with open('total-transactions-per-mode.png', 'rb') as imagefile:
                chart6 = base64.b64encode(imagefile.read()).decode('ascii')
                chart6 = "data:image/png;base64,"+chart6
            with open('transaction-modes-monthly.png', 'rb') as imagefile:
                chart7 = base64.b64encode(imagefile.read()).decode('ascii')
                chart7 = "data:image/png;base64,"+chart7
            with open('gaugex.png', 'rb') as imagefile:
                chart8 = base64.b64encode(imagefile.read()).decode('ascii')
                chart8 = "data:image/png;base64,"+chart8
            return Response({'Monthly_investment_required':int(principal),
            'Average_savings_per_month':int(avg_savings),
            'Amount_gained_due_to_RD':int(profit),
            'Goal_achieveable':purchase_possible,
            'GST_Score':score,
            'fd_principal':int(fd_principal),
            'spending_patterns':chart1,
            #'monthly_total_transactions':chart2,
            'monthly_debit_credit':chart3,
            'monthy_savings':chart4,
            'savings_vs_rd':chart5,
            'total_transactions_per_mode':chart6,
            'transaction_modes_monthly':chart7,
            'GST_Speedometer':chart8
            })
        else:
            return Response(serializer.errors, status = status.HTTP_400_BAD_REQUEST)
