"use client";

import React, { useRef, useState } from "react";
import Webcam from "react-webcam";
import axios from "axios";
import "bootstrap/dist/css/bootstrap.min.css";

const FacialMapping = ({ patient_name, onComplete }) => {
  const [imageData, setImageData] = useState("");
  const [response, setResponse] = useState(null);
  const [error, setError] = useState("");
  const [submittedImage, setSubmittedImage] = useState(false);
  const webcamRef = useRef(null);

  const handleCapture = () => {
    const imageSrc = webcamRef.current?.getScreenshot();
    setImageData(imageSrc);
  };

  const handleSubmit = async () => {
    setError("");
    setResponse(null);

    if (!imageData) {
      setError("Please capture an image first.");
      return;
    }

    try {
      const res = await axios.post("http://localhost:5000/analysis/facial-mapping", {
        name: patient_name,
        image: imageData.split(",")[1],
      });

      setResponse(res.data);
      setSubmittedImage(true);

      const resultPayload = {
        patient_name: patient_name,
        result:  res.data.significant_change? "Significant change detected": "No significant change detected.",
      };

      await axios.post("http://localhost:5000/facial-analysis/save-results", resultPayload, {
        headers: {
          'Content-Type': 'application/json',
        }
      });
    } catch (err) {
      setError(err.response?.data?.error || "An error occurred.");
    }
  };

  return (
    <div className="container mt-4">
      <h2 className="text-center mb-4">Facial Mapping</h2>

      <div className="text-center mb-4">
        <h5>Take a Picture</h5>
        <Webcam
          audio={false}
          ref={webcamRef}
          screenshotFormat="image/png"
          className="border rounded shadow-sm"
          style={{ width: "100%", maxWidth: "500px" }}
        />
        <div className="mt-3">
          <button
            className="btn btn-secondary me-2"
            onClick={handleCapture}
            disabled={submittedImage}
          >
            Capture Image
          </button>
        </div>
      </div>

      {imageData && (
        <div className="text-center mb-4">
          <h5>Preview</h5>
          <img
            src={imageData}
            alt="Preview"
            className="img-fluid border rounded shadow-sm"
            style={{ maxHeight: "300px", objectFit: "contain" }}
          />
        </div>
      )}

      <div className="d-flex justify-content-center gap-3 mb-4">
        <button
          className="btn btn-primary"
          onClick={handleSubmit}
          disabled={submittedImage}
        >
          Submit
        </button>
        <button
          className="btn btn-success"
          onClick={onComplete}
          disabled={!submittedImage}
        >
          Continue
        </button>
      </div>

      {error && (
        <div className="alert alert-danger text-center">
          <strong>Error:</strong> {error}
        </div>
      )}

      {response && (
        <div className="card mt-3 shadow-sm">
          <div className="card-body text-center">
            <h5 className="card-title">Analysis Results</h5>
            <p className="card-text">
              <strong>Message:</strong> {response.message}
            </p>
            {response.significant_change && (
              <p className="card-text text-danger">
                <strong>ALERT:</strong> Significant facial changes detected!
              </p>
            )}
            <p className="card-text">
              <strong>Change Value:</strong>{" "}
              {isNaN(response.change_value) ? "N/A" : response.change_value.toFixed(2)}
            </p>
            {response.annotated_image && (
              <img
                src={`data:image/png;base64,${response.annotated_image}`}
                alt="Annotated"
                className="img-fluid border rounded shadow-sm mt-3"
                style={{ maxHeight: "300px", objectFit: "contain" }}
              />
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default FacialMapping;
