# myapp/forms.py
from django import forms

class QuestionForm(forms.Form):
    question = forms.CharField(label='Ask a question', max_length=255, widget=forms.TextInput(attrs={'placeholder': 'Ask your question...'}))
