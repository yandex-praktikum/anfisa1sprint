from django.shortcuts import render


def index(request):
    template = 'homepage/index.html'
    return render(request, template)
