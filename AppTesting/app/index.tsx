import React, { useState } from "react";
import { View, Text, Button, StyleSheet, Platform, Alert } from "react-native";

interface videoOutput {
    file: any;
    text: string;
    plots: Plot[];
}
interface Plot {
    imageBase64: string;
    mimeType: string;
}



export default function VideoUploader() {
    const [videoFile, setVideoFile] = useState(Array<any>);
    const [videoURL, setVideoURL] = useState(["", ""]);
    const [loading, setStatus] = useState(Boolean);
    const [output, setOutput] = useState<videoOutput>({
        file: null,
        text: "placeholder",
        plots: [],
    });
    const [outputURL, setOutputURL] = useState<string | null>(null);

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
        const formData = new FormData();
        for (let i = 0; i < videoFile.length; i++) {
            if (videoFile[i] != null) {
                formData.append(`file${i + 1}`, videoFile[i]);
            }
        }
        setStatus(true);
        try {
            const response = await fetch("http://localhost:5000/upload", {
                method: "POST",
                body: formData,
            });

            if (response.ok) {
                const data = await response.json();
                // data.textCorrections: array of strings
                // data.videoBase64: base64 string
                // data.videoMimeType: e.g. "video/mp4"

                // Convert base64 to Blob
                const byteCharacters = atob(data.videoBase64);
                const byteNumbers = new Array(byteCharacters.length);
                for (let i = 0; i < byteCharacters.length; i++) {
                    byteNumbers[i] = byteCharacters.charCodeAt(i);
                }
                const byteArray = new Uint8Array(byteNumbers);
                const videoBlob = new Blob([byteArray], { type: data.videoMimeType || "video/mp4" });

                let curr_output: videoOutput = {
                    file: videoBlob,
                    text: Array.isArray(data.textCorrections) ? data.textCorrections.join("\n") : String(data.textCorrections),
                    plots: data.plots,
                };
                setOutput(curr_output);
                console.log("Output:", curr_output);

                if (videoBlob) {
                    const url = URL.createObjectURL(videoBlob);
                    setOutputURL(url);
                }
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
        <div>
            {output.file == null && (
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
                                    <button
                                        onClick={(event) =>
                                            DeleteFile(event, 0)
                                        }
                                    >
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
                                    <button
                                        onClick={(event) =>
                                            DeleteFile(event, 1)
                                        }
                                    >
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
                    {
                        <Button
                            title={loading ? "Uploading..." : "Upload Video"}
                            onPress={UploadVideo}
                        />
                    }
                </View>
            )}
            {output.file != null && outputURL && (
                <View style={styles.upload_container}>
                    <Text style={styles.title}>Output</Text>
                    <video
                        width="600"
                        height="400"
                        controls
                        src={outputURL}
                    >
                        Your browser does not support the video tag.
                    </video>
                    <a
                        href={outputURL}
                        download="output_video.mp4"
                        style={{ marginTop: 16, display: "inline-block", fontSize: 18 }}
                    >
                        Download Video
                    </a>
                    <p>{output.text}</p>

                    <div>
                        <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
    {output.plots.map((plot, index) => (
        <img
            key={index}
            src={`data:${plot.mimeType};base64,${plot.imageBase64}`}
            alt={`Plot ${index + 1}`}
            style={{ maxWidth: 1000, marginBottom: 20 }}
        />
    ))}
</div>
                    </div>
                </View>
            )}
        </div>
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
    output_text: {
        fontSize: 20,
        margin: 20,
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
