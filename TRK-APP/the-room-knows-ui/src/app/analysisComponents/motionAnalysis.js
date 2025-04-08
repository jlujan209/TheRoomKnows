"use client";

import React, { useRef, useState, useEffect } from "react";
import Webcam from "react-webcam";
import axios from "axios";
import "bootstrap/dist/css/bootstrap.min.css";
import { useRouter } from "next/navigation";


const MotionAnalysis = ({ patient_name, onComplete }) => {
    const [response, setResponse] = useState(null);
    const [error, setError] = useState("");
    const [analyzing, setAnalyzing] = useState(false);
    const webcamRef = useRef(null);
    let intervalRef = useRef(null);
    const router = useRouter();

    const captureImage = async () => {
        if (webcamRef.current) {
          const imageSrc = webcamRef.current.getScreenshot();
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
          <h2 className="mb-3">Motion Analysis</h2>
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
              {analyzing ? "Stop Recording" : "Start Recording"}
            </button>
            <button className="btn btn-primary" onClick={onComplete}>continue</button>
          </div>
    
          {error && (
            <div className="alert alert-danger mt-3">
              <strong>Error:</strong> {error}
            </div>
          )}
    
        </div>
      );
}

export default MotionAnalysis;