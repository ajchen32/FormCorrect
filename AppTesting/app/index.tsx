import { Text, View,  StyleSheet } from 'react-native';
import Button from '@/components/button';
import VideoScreen from '@/components/videoplayer'
import * as ImagePicker from 'expo-image-picker';
// import {downloadAsync } from 'expo-file-system';

let videoSrc : string = ''; //URI of the file chosen

export default function Index() {
  const pickImageAsync = async () => {
    let result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['videos'],
      allowsEditing: true,
      quality: 1,
    });

    if (!result.canceled) {
      console.log(result.assets[0].uri);
      videoSrc = result.assets[0].uri;

    } else {
      alert('You did not select any files.');
    }
  };

  const printVideoSrc = () =>{
    console.log(videoSrc);
  }

  return (
    <View style={styles.container}>
      <Text style={styles.text}>What am I doing</Text>
      <View style={styles.imageContainer}>
        <VideoScreen uriIn={videoSrc}/>
      </View>
      <View style={styles.footerContainer}>
        <Button theme="primary" label="Upload a video" onPress={pickImageAsync}/>
        <Button theme="primary" label="Use this video" onPress={printVideoSrc}/>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#25292e',
    alignItems: 'center',
    justifyContent: 'center',
  },
  text: {
    color: '#fff',
  },
  imageContainer: {
    flex: 1,
  },
  footerContainer: {
    flex: 1 / 3,
    alignItems: 'center',
  },
});
