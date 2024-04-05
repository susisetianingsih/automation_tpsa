REM Build Docker image
docker build -t my-api-image .

REM Run Docker container
docker run -d -p 5000:5000 my-api-image
