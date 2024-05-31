from django.shortcuts import render


def ProfilerForm(request):
    possible_department = ['N/A','Audit', 'Credit&Coll', 'Csp', 'Financial', 'GRCD', 'Hr&Admin', 'M2', 'Management Accounting', 'MIS', 'Operation','PSPMI', 'Transportation Lisence', 'SDOG' ]
    
    return render(request, 'myapp/ProfilerForm.html',{'possible_department':possible_department})