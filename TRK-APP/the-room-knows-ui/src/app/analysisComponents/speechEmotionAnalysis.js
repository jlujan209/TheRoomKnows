"use client";

import React, { useRef, useState, useEffect } from "react";
import Webcam from "react-webcam";
import axios from "axios";
import "bootstrap/dist/css/bootstrap.min.css";
import { useRouter } from "next/navigation";
import io from "socket.io-client";

const SpeechEmotionAnalysis = ({patient_name, onComplete, emotion, speech}) => {

        const [response, setResponse] = useState(null);
        const [error, setError] = useState("");
        const [analyzing, setAnalyzing] = useState(false);
        const webcamRef = useRef(null);
        let intervalRef = useRef(null);
        const router = useRouter();
        const socketRef = useRef(null);

        useEffect(() => {
            return () => {
              clearInterval(intervalRef.current);
              if (socketRef.current) {
                socketRef.current.disconnect();
              }
            };
          }, []);

        const [emotionCounter, setEmotionCounter] = useState({
            Angry: 0,
            Happy: 0,
            Neutral: 0,
            Sad: 0,
            Surprise: 0,
          });
      
        const captureImage = async () => {
          if (webcamRef.current) {
            const imageSrc = webcamRef.current.getScreenshot();
            
            try {
              const res = await axios.post("https://localhost:5000/analysis/emotion-detection", {
                image: imageSrc.split(",")[1],
              });
              setResponse(res.data);
              
              setEmotionCounter(prev => ({
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
            socketRef.current = io('http://localhost:5000');

            socketRef.current.on("connect", ()=>{
                console.log("Socket connected for Speech Analysis");
            })
          }
        };
      
        const stopAnalysis = () => {
          setAnalyzing(false);
          if (emotion) {
            clearInterval(intervalRef.current);
          }
          if (speech && socketRef.current) {
            socketRef.current.disconnect();
            socketRef.current = null;
          }
        };
      
        useEffect(() => {
          return () => clearInterval(intervalRef.current);
        }, []);

        const complete = async() => {
            const emotionDataString = JSON.stringify(emotionCounter);
            console.log(emotionDataString);

            const emotion_response = await fetch('http://localhost:5000/analysis/emotion-detection/save-results', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: {
                    patient_name: patient_name,
                    results: emotionDataString,
                }
            })

            const result = await emotion_response.json();
            console.log(result);

            onComplete();
        }

        if (emotion) {  

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
                <button className='btn btn-primary' onClick={complete} disabled={analyzing}>Continue</button>
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
                    
                    </div>
                </div>
                )}
            </div>
            );
        } else {
            return (
                <>
                    <h4>Speech Analysis</h4>
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
                        <button className='btn btn-primary' onClick={onComplete} disabled={analyzing}>Continue</button>
                        </div>
                </>
            );
        }
}

export default SpeechEmotionAnalysis;