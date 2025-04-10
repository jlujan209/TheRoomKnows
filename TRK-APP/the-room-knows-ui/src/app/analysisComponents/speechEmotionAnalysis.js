"use client";

import React, { useRef, useState, useEffect } from "react";
import Webcam from "react-webcam";
import axios from "axios";
import "bootstrap/dist/css/bootstrap.min.css";
import { useRouter } from "next/navigation";
import io from "socket.io-client";

const SpeechEmotionAnalysis = ({ patient_name, onComplete, emotion, speech }) => {
  const [response, setResponse] = useState(null);
  const [error, setError] = useState("");
  const [analyzing, setAnalyzing] = useState(false);
  const [emotionCounter, setEmotionCounter] = useState({
    Angry: 0,
    Happy: 0,
    Neutral: 0,
    Sad: 0,
    Surprise: 0,
  });

  const webcamRef = useRef(null);
  const intervalRef = useRef(null);
  const socketRef = useRef(null);
  const router = useRouter();

  // Cleanup
  useEffect(() => {
    return () => {
      clearInterval(intervalRef.current);
      if (socketRef.current) socketRef.current.disconnect();
    };
  }, []);

  // Capture and send image for emotion analysis
  const captureImage = async () => {
    if (webcamRef.current) {
      const imageSrc = webcamRef.current.getScreenshot();
      try {
        const res = await axios.post("http://localhost:5000/analysis/emotion-detection", {
          image: imageSrc.split(",")[1],
        });

        setResponse(res.data);
        setEmotionCounter((prev) => ({
          ...prev,
          [res.data.emotion]: prev[res.data.emotion] + 1,
        }));
      } catch (err) {
        setError(err.response?.data?.error || "An error occurred");
      }
    }
  };

  const startAnalysis = () => {
    setAnalyzing(true);
    setError("");
    setResponse(null);

    if (emotion) {
      intervalRef.current = setInterval(captureImage, 5000);
    }

    if (speech) {
      socketRef.current = io("http://localhost:5000", {
        query: {
          patient_name: patient_name, // This will get sent with the initial connection
        },
      });
      socketRef.current.on("connect", () => {
        console.log("Socket connected for Speech Analysis");
      });
    }
  };

  const stopAnalysis = () => {
    setAnalyzing(false);
    if (emotion) clearInterval(intervalRef.current);
    if (speech && socketRef.current) {
      socketRef.current.disconnect();
      socketRef.current = null;
    }
  };

  const complete = async () => {
    try {
      const res = await fetch("http://localhost:5000/analysis/emotion-detection/save-results", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          patient_name,
          results: emotionCounter,
        }),
      });

      const data = await res.json();
      console.log(data);
    } catch (err) {
      console.error("Error saving results:", err);
    }
    onComplete();
  };

  return (
    <div className="container mt-4 text-center">
      <h2 className="mb-3">
        {emotion ? "Real-Time Emotion Detection" : "Speech Analysis"}
      </h2>

      {emotion && (
        <div className="d-flex justify-content-center">
          <Webcam
            audio={false}
            ref={webcamRef}
            screenshotFormat="image/png"
            className="border rounded"
            style={{ width: "100%", maxWidth: "500px" }}
          />
        </div>
      )}

      <div className="d-flex justify-content-center gap-3 mt-4 flex-wrap">
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

        <button
          className="btn btn-success"
          onClick={emotion ? complete : onComplete}
          disabled={analyzing}
        >
          Continue
        </button>
      </div>

      {error && (
        <div className="alert alert-danger mt-3">
          <strong>Error:</strong> {error}
        </div>
      )}

      {response && emotion && (
        <div className="card mt-4 shadow-sm mx-auto" style={{ maxWidth: "500px" }}>
          <div className="card-body">
            <h5 className="card-title">Latest Detection</h5>
            <p className="card-text">
              <strong>Emotion:</strong> {response.emotion}
            </p>
            <p className="card-text">
              <strong>Confidence:</strong>{" "}
              {isNaN(response.confidence) ? "N/A" : response.confidence.toFixed(2)}
            </p>
          </div>
        </div>
      )}
    </div>
  );
};

export default SpeechEmotionAnalysis;
