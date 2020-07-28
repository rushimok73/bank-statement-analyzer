from rest_framework import serializers

class scriptserializer(serializers.Serializer):
    goal = serializers.IntegerField()
    years = serializers.IntegerField()
