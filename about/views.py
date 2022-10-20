from django.shortcuts import render


def description(request):
    template = 'about/description.html'
    return render(request, template)
