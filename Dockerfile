FROM python:3.8
ADD requirements.txt /
ADD ./* /
RUN pip install -r /requirements.txt
CMD ["python", "./toxicityPrediction.py"]