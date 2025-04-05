import React, { useState } from 'react';
import { View, Text, Button, StyleSheet, Platform, Alert } from 'react-native';

export default function VideoUploader() {
  const [videoFile, setVideoFile] = useState(null);
  const [videoURL, setVideoURL] = useState('');
  const [loading, setStatus] = useState(Boolean);

  // Do something if selected a different file
  const handleFileChange = (event : any) => {
    if (event){ //Add  [&& event.currentTarget.files[0].type == 'video/mp4'] to enforce a certain type
      setVideoFile(event.currentTarget.files[0]);
      setVideoURL(URL.createObjectURL(event.currentTarget.files[0]));
      setStatus(false);
    }
  };

  const UploadVideo = async () => {
    // Create a FormData object to send the file
    const formData = new FormData();
    if (videoFile != null){
      formData.append('file', videoFile);
    }
    setStatus(true);
    try {
      const response = await fetch('http://localhost:5000/upload', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        alert('File uploaded successfully!');
      } else {
        alert('Failed to upload file.');
      }
    } catch (error) {
      console.error('Error uploading file:', error);
      alert('Error uploading file.');
    }
    setStatus(false);
  };

  
  return (
    <View style={styles.container}>
      <Text style={styles.title}>Upload a Video</Text>
      <input type="file" onChange={handleFileChange} />
        <View style={styles.videoContainer}>
          {videoURL && (
          <video 
            width="600" 
            height="400" 
            controls 
            src={videoURL} // Set the video source to the URL created with createObjectURL
          >
            Your browser does not support the video tag.
          </video>)}

          {videoFile && (
            <Button
              title={loading ? 'Uploading...' : 'Upload Video'}
              onPress={UploadVideo}/>)}

        </View>
    </View>
  );
}
 
const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 16,
  },
  title: {
    fontSize: 24,
    marginBottom: 20,
  },
  videoContainer: {
    marginTop: 20,
    alignItems: 'center',
  },
  video: {
    width: 300,
    height: 300,
  },
});

