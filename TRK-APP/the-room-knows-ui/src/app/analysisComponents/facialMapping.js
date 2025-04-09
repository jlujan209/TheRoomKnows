"use client";

import React, { useRef, useState, useEffect } from "react";
import Webcam from "react-webcam";
import "bootstrap/dist/css/bootstrap.min.css";

const FacialMapping = ({ patient_name, onComplete }) => {
  const [imageData, setImageData] = useState("");
  const [patientName, setPatientName] = useState("");
  const [response, setResponse] = useState(null);
  const [error, setError] = useState("");
  const webcamRef = useRef(null);
  const [ submittedImage, setSubmittedImage ] = useState(false);

  const handleCapture = () => {
    const imageSrc = webcamRef.current.getScreenshot();
    setImageData(imageSrc);
  };

  const handleUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setImageData(e.target.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleSubmit = async () => {
    setError("");
    setResponse(null);

    if (!patientName) {
      setError("Patient name is required.");
      return;
    }

    if (!imageData) {
      setError("Please capture or upload an image.");
      return;
    }

    try {
      const response = await axios.post("http://localhost:5000/analysis/facial-mapping", {
        name: patient_name,
        image: imageData.split(",")[1], // Remove Base64 prefix
      });

      setResponse(response.data);
      setSubmittedImage(true);
    } catch (err) {
      setError(err.response?.data?.error || "An error occurred.");
    }
  };

  return (
    <div style={{ padding: "20px" }}>
      <h1>Facial Mapping App</h1>

      <div>
        <label>Patient Name:</label>
        <input
          type="text"
          value={patientName}
          onChange={(e) => setPatientName(e.target.value)}
          placeholder="Enter patient name"
        />
      </div>

      <div style={{ marginTop: "20px" }}>
        <h2>Take a Picture:</h2>
        <Webcam
          audio={false}
          ref={webcamRef}
          screenshotFormat="image/png"
          style={{ width: "100%", maxHeight: "300px" }}
        />
        <button onClick={handleCapture} disabled={submittedImage}>Capture Image</button>
      </div>

      {imageData && (
        <div style={{ marginTop: "20px" }}>
          <h3>Preview:</h3>
          <img
            src={imageData}
            alt="Preview"
            style={{
              width: "100%", 
              height: "auto",   // Maintains the aspect ratio
              maxHeight: "300px",  // Maximum height
              objectFit: "contain" // Ensures the image maintains its aspect ratio within the container
            }}
          />
        </div>
      )}

      <div style={{ marginTop: "20px" }}>
        <button onClick={handleSubmit} disabled={submittedImage} className="btn btn-primary">Submit</button>
        <button onClick={onComplete} disabled={!submittedImage} className="btn btn-primary">Continue</button>
      </div>

      {error && (
        <div style={{ marginTop: "20px", color: "red" }}>
          <strong>Error:</strong> {error}
        </div>
      )}

    {response && (
      <div style={{ marginTop: "20px" }}>
        <h3>Results:</h3>
        <p><strong>Message:</strong> {response.message}</p>
        {response.significant_change && (
          <p><strong>ALERT:</strong> Significant facial changes detected!</p>
        )}
        <p>
          <strong>Change Value:</strong> 
          {isNaN(response.change_value) ? "N/A" : response.change_value.toFixed(2)}
        </p>
        {response.annotated_image && (
          <img
            src={`data:image/png;base64,${response.annotated_image}`}
            alt="Annotated"
            style={{
              width: "100%", 
              height: "auto",   // Maintains the aspect ratio
              maxHeight: "300px",  // Maximum height
              objectFit: "contain"
            }}
          />
        )}
      </div>
    )}

    </div>
  );
}

export default FacialMapping;