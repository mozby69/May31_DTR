from django.shortcuts import render


def Profiler(request):
    return render(request, 'myapp/Profiler.html')