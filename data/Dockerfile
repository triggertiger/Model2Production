FROM python:3.9-slim-bullseye

COPY . /database/

RUN pip install -r database/requirements_docker.txt
 
WORKDIR /database

# Give execution permissions 
RUN chmod +x set_database.sh

# Set bash as entrypoint
ENTRYPOINT [ "/bin/bash", "/database/set_database.sh"]

