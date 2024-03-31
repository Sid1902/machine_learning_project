FROM python:3.7 
COPY . /app 
WORKDIR /app
RUN pip install -r requirements.txt
RUN pip install gunicorn
EXPOSE $PORT 

# This line exposes a port on the container. 
# The $PORT variable refers to an environment variable that you can set when running the container image. 
# This allows you to map the exposed port on the container to a specific port on your host machine.
CMD gunicorn --workers=1 --bind 0.0.0.0:$PORT app:app

# It uses the gunicorn web server to serve your Python application.

# --bind 0.0.0.0:$PORT: Binds Gunicorn to listen on all interfaces (0.0.0.0) on the port specified by the $PORT environment variable.

# app:app: This tells Gunicorn the application entry point . 
#  it assumes your application is in a file named app.py and the function to serve is called app