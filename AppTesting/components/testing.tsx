// A temp file to store old code while I test other stuff

import React, { useState } from 'react';
import { View, Text, Button, StyleSheet, Platform, Alert } from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { Video } from 'expo-av';
import * as FileSystem from 'expo-file-system';
import axios from 'axios'; // You can also use 'fetch' if you prefer
let videoSrc : string = ''; //URI of the file chosen

export default function VideoUploader() {
  const [videoUri, setVideoUri] = useState(String);
  const [loading, setLoading] = useState(false);
  let selectedFile : File;

  const checkOS = async () => {

  }

  // Request permission for media access
  const requestPermissions = async () => {
    if (Platform.OS != 'web') {
      const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
      if (status !== 'granted') {
        alert('Sorry, we need camera roll permissions to pick videos!');
      }
    }
    else{
    }
  };

  // WEB------------------------
  const handleFileChange = (event : any) => {
    selectedFile = event.target.files[0];
    console.log(selectedFile);
    
  };

  //Working on this right now------------
  const webUploadVideo = async () => {
    const formData = new FormData();

    formData.append('video', selectedFile);
    // You can use either fetch or axios for making the request
    const response = await fetch('/upload', { // Replace with your backend endpoint
      method: 'POST',
      body: formData,
    });

    if (response.status === 200) {
      Alert.alert('Success', 'Video uploaded successfully!');
    } else {
      Alert.alert('Error', 'Failed to upload the video');
    }
  }

  // IOS/ANDRIOID MAYBE--------------------------
  // Function to handle video selection
  const pickVideo = async () => {
    let result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['videos'],
      allowsEditing: true,
      quality: 1,
    });

    if (!result.canceled) {
      console.log(result.assets[0].uri);
      setVideoUri(result.assets[0].uri); }
    }

  // Function to upload the video to the server
  const uploadVideo = async () => {
    if (!videoUri) {
      alert('Please select a video first!');
      return;
    }

    setLoading(true);

    try {
        // Prepare the form data for the video upload
      const formData = new FormData();
      formData.append('video', new Blob([videoUri], {type: 'video/mp4'}));

      const response = await fetch('http://localhost:5000/upload', {
        method: 'POST',
        body: formData, // Form data containing the video
        headers: {
          // Do not manually set Content-Type; let fetch do it automatically
        },
      });
      // Check if the request was successful
      if (response.ok) {
        Alert.alert('Success', 'Video uploaded successfully!');
      } else {
        const errorMessage = await response.text();
        Alert.alert('Error', errorMessage || 'Failed to upload the video');
      }
      // }
      
    } catch (error) {
      console.error('Upload failed:', error);
      Alert.alert('Error', 'Failed to upload the video');
    } finally {
      setLoading(false);
    }
  };

  // No clue what this is
  React.useEffect(() => {
    requestPermissions();
  }, []);

  // Check the current OS
  if (Platform.OS == 'web'){
    return (
      <View style={styles.container}>
        <Text style={styles.title}>Upload a Video</Text>
        <input type="file" onChange={handleFileChange} />
        {videoUri && (
          <View style={styles.videoContainer}>
            <Video
              source={{ uri: videoUri }}
              style={styles.video}
              useNativeControls
              isLooping
            />
          </View>
        )}
        {(
          <Button
            title={loading ? 'Uploading...' : 'Upload Video'}
            onPress={webUploadVideo}
            disabled={loading}
          />
        )}
      </View>
    );
  }
  else{
    return (
      <View style={styles.container}>
        <Text style={styles.title}>Upload a Video</Text>
        <Button title="Pick a Video" onPress={pickVideo} />
        {videoUri && (
          <View style={styles.videoContainer}>
            <Video
              source={{ uri: videoUri }}
              style={styles.video}
              useNativeControls
              isLooping
            />
          </View>
        )}
        {videoUri && (
          <Button
            title={loading ? 'Uploading...' : 'Upload Video'}
            onPress={uploadVideo}
            disabled={loading}
          />
        )}
      </View>
    );
  }
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

