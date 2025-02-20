"use client";

import React, { useRef, useState } from "react";
import Webcam from "react-webcam";
import axios from "axios";

const WebcamCapture = () => {
  const [ imageData, setImageData ] = useState("");
  const [ response, setResponse ] = useState("");
  const [ error, setError ] = useState("");
  const webcamRef = useRef(null);

  const handleCapture = () => {
    const imageSrc = webcamRef.current.getScreenshot();
    setImageData(imageSrc);
  }

  const handleSubmit = async () => {
    setError("");
    setResponse(null);
     if (!imageData) {
      setError("Please capture an image");
      return;
     }

     try {
      const response = await axios.post("https://localhost:5000/analysis/emotion-detection", {
        image: imageData.split(",")[1],
      });

      setResponse(response.data);
     } catch (err) {
      setError(err.response?.data?.error || "An error occured");
     }
  }

  return (
    <>
      <Webcam 
      audio={false}
      ref={webcamRef} 
      screenshotFormat="image/png"
      style={{width: "100%", maxHeight: "300px"}}
      />
      <button onClick={handleCapture}>Capture</button>
      {imageData && (
        <div style={{ marginTop: "20px" }}>
          <h3>Preview:</h3>
          <img
            src={imageData}
            alt="Preview"
            style={{
              width: "100%", 
              height: "auto",   
              maxHeight: "300px",  
              objectFit: "contain" 
            }}
          />
        </div>
      )}
      <button onClick={handleSubmit}>Submit</button>

      {error && (
        <div style={{ marginTop: "20px", color: "red" }}>
          <strong>Error:</strong> {error}
        </div>
      )}

      {response && (
      <div style={{ marginTop: "20px" }}>
        <h3>Results:</h3>
        <p><strong>Emotion:</strong> {response.emotion}</p>
        <p>
          <strong>Confidence:</strong> 
          {isNaN(response.confidence) ? "N/A" : response.confidence.toFixed(2)}
        </p>
        {response.annotated_image && (
          <img
            src={`data:image/png;base64,${response.annotated_image}`}
            alt="Annotated"
            style={{
              width: "100%", 
              height: "auto",   
              maxHeight: "300px",  
              objectFit: "contain"
            }}
          />
        )}
      </div>
    )}
    </>
  )
};

export default WebcamCapture;
