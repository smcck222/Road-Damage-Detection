# Road Damage Detection

Dockerized Flask App to segment cracks and detect potholes on road images. 

**To create docker image of the project:**

sudo docker build -t app .

**To run the image created.**

sudo docker run --name app -p 5000:5000 app

The containter is up and running on port 5000. 
Use the CURL command to send an image as a POST request to the Flask App running on localhost. 
The result is a JPEG image. 

- **For Cracks** : 

curl -k -X POST -F 'image=@image_path/ -v http://0.0.0.0:5000/segment > crack_result.jpeg 
- **For Potholes** : 

curl -k -X POST -F 'image=@image_path/ -v http://0.0.0.0:5000/detect/rcnn > pothole_result.jpeg

 
