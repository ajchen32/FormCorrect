import React, { useState } from "react";
import { View, Text, Button, StyleSheet, Platform, Alert } from "react-native";

export default function VideoUploader() {
    const [videoFile, setVideoFile] = useState(Array<any>);
    const [videoURL, setVideoURL] = useState(["", ""]);
    const [loading, setStatus] = useState(Boolean);

    // Do something if selected a different file
    const LoadFile = (event: any, index: number) => {
        if (event) {
            //Add  [&& event.currentTarget.files[0].type == 'video/mp4'] to enforce a certain type
            let vidFile = [...videoFile];
            let vidURL = [...videoURL];

            vidFile[index] = event.currentTarget.files[0];
            vidURL[index] = URL.createObjectURL(event.currentTarget.files[0]);

            setVideoFile(vidFile);
            setVideoURL(vidURL);
            console.log(vidFile, vidURL);
        }
    };

    const DeleteFile = (event: any, index: number) => {
        if (event) {
            let vidFile = [...videoFile];
            let vidURL = [...videoURL];
            vidFile[index] = null;
            vidURL[index] = "";
            setVideoFile(vidFile);
            setVideoURL(vidURL);
            console.log(vidFile, vidURL);
        }
    };

    const UploadVideo = async () => {
        // Create a FormData object to send the file
        const formData = new FormData();
        for (let i = 0; i < videoFile.length; i++) {
            if (videoFile[i] != null) {
                formData.append(`file${i+1}`, videoFile[i]);
            }
        }
        setStatus(true);
        try {
            const response = await fetch("http://localhost:5000/upload", {
                method: "POST",
                body: formData,
            });

            if (response.ok) {
                alert(`File uploaded successfully!`);
            } else {
                alert(`Failed to upload file.`);
            }
        } catch (error) {
            console.error(`Error uploading file:`, error);
            alert(`Error uploading file.`);
        }
        setStatus(false);
    };

    return (
        <View style={styles.upload_container}>
            <Text style={styles.title}>Upload a Video</Text>

            <View style={styles.container}>
                <View style={styles.videoContainer}>
                    <View style={styles.upload_button}>
                        <input
                            type="file"
                            accept="video/*"
                            onChange={(event) => LoadFile(event, 0)}
                        />
                        {videoURL[0] != "" && (
                            <button onClick={(event) => DeleteFile(event, 0)}>
                                Delete
                            </button>
                        )}
                    </View>

                    {videoURL[0] && (
                        <video
                            width="600"
                            height="400"
                            controls
                            src={videoURL[0]} // Set the video source to the URL created with createObjectURL
                        >
                            Your browser does not support the video tag.
                        </video>
                    )}
                </View>

                <View style={styles.videoContainer}>
                    <View style={styles.upload_button}>
                        <input
                            type="file"
                            accept="video/*"
                            onChange={(event) => LoadFile(event, 1)}
                        />
                        {videoURL[1] != "" && (
                            <button onClick={(event) => DeleteFile(event, 1)}>
                                Delete
                            </button>
                        )}
                    </View>

                    {videoURL[1] && (
                        <video
                            width="600"
                            height="400"
                            controls
                            src={videoURL[1]} // Set the video source to the URL created with createObjectURL
                        >
                            Your browser does not support the video tag.
                        </video>
                    )}
                </View>
            </View>

            <Button
                title={loading ? "Uploading..." : "Upload Video"}
                onPress={UploadVideo}
            />
        </View>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        justifyContent: "center",
        alignItems: "center",
        display: "flex",
        flexDirection: "row",
        padding: 16,
    },
    title: {
        fontSize: 24,
        marginBottom: 20,
        justifyContent: "center",
    },
    upload_button: {
        padding: 16,
        display: "flex",
        flexDirection: "row",
    },
    videoContainer: {
        alignItems: "center",
        display: "flex",
        flexDirection: "column",
        padding: 16,
    },
    video: {
        width: 300,
        height: 300,
    },
    upload_container: {
        justifyContent: "center",
        alignItems: "center",
        display: "flex",
        flexDirection: "column",
        padding: 16,
    },
});
