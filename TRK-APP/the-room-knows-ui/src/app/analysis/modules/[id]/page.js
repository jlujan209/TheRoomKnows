"use client";

import React, { useRef, useState, useEffect } from "react";
import Webcam from "react-webcam";
import axios from "axios";
import "bootstrap/dist/css/bootstrap.min.css";
import withAuth from "../../../hoc/withAuth";
import { useRouter } from "next/navigation";

const WebcamCapture = () => {
  const [response, setResponse] = useState(null);
  const [error, setError] = useState("");
  const [analyzing, setAnalyzing] = useState(false);
  const webcamRef = useRef(null);
  let intervalRef = useRef(null);
  const router = useRouter();

  const captureImage = async () => {
    if (webcamRef.current) {
      const imageSrc = webcamRef.current.getScreenshot();
      
      try {
        const res = await axios.post("http://localhost:5000/analysis/emotion-detection", {
          image: imageSrc.split(",")[1],
        });
        setResponse(res.data);
      } catch (err) {
        setError(err.response?.data?.error || "An error occurred");
      }
    }
  };

  const startAnalysis = () => {
    setAnalyzing(true);
    setError("");
    setResponse(null);
    intervalRef.current = setInterval(captureImage, 5000);
  };

  const stopAnalysis = () => {
    setAnalyzing(false);
    clearInterval(intervalRef.current);
  };

  useEffect(() => {
    return () => clearInterval(intervalRef.current);
  }, []);

  return (
    <div className="container mt-4 text-center">
      <h2 className="mb-3">Real-Time Emotion Detection</h2>
      <div className="d-flex justify-content-center">
        <Webcam
          audio={false}
          ref={webcamRef}
          screenshotFormat="image/png"
          className="border rounded"
          style={{ width: "100%", maxWidth: "500px" }}
        />
      </div>

      <div className="d-flex justify-content-center gap-3 mt-4">
        <button 
        type="button" 
        onClick={() => router.back()} 
        className="btn btn-secondary"
        disabled={analyzing}
        >
          Cancel
        </button>
        <button 
          className={`btn ${analyzing ? "btn-danger" : "btn-primary"}`}  
          onClick={analyzing ? stopAnalysis : startAnalysis}
        >
          {analyzing ? "Stop Analysis" : "Start Analysis"}
        </button>
      </div>

      {error && (
        <div className="alert alert-danger mt-3">
          <strong>Error:</strong> {error}
        </div>
      )}

      {response && (
        <div className="card mt-3 p-3 text-center shadow-sm">
          <div className="card-body">
            <h5 className="card-title">Analysis Results</h5>
            <p className="card-text">
              <strong>Emotion:</strong> {response.emotion}
            </p>
            <p className="card-text">
              <strong>Confidence:</strong> {isNaN(response.confidence) ? "N/A" : response.confidence.toFixed(2)}
            </p>
            {response.annotated_image && (
              <img
                src={`data:image/png;base64,${response.annotated_image}`}
                alt="Annotated"
                className="img-fluid mt-2 border rounded"
                style={{ maxWidth: "100%", maxHeight: "300px" }}
              />
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default withAuth(WebcamCapture);
