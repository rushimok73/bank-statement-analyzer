from rest_framework import serializers

class scriptserializer(serializers.Serializer):
    goal = serializers.IntegerField()
    date = serializers.CharField()
    AA_ID = serializers.CharField()
    FI_data = serializers.CharField()
