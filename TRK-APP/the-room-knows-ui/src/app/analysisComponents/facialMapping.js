"use client";

import React, { useRef, useState, useEffect } from "react";
import Webcam from "react-webcam";
import "bootstrap/dist/css/bootstrap.min.css";

const FacialMapping = ({ patient_name, onComplete }) => {

    const [ loading, setLoading ] = useState(false);
    const [ response, setResponse ] = useState(null);
    const [ error, setError ] = useState(null);
    const [ annotatedImage, setAnnotatedImage ] = useState(null);
    const webcamRef = useRef(null);

    const handleCapture = async () => {
        setLoading(true);
        setError(null);
        setResponse(null);
        setAnnotatedImage(null);

        try {
            const res = await fetch("http://localhost:5000/analysis/facial-mapping", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ name: patient_name})
            });

            const data = await res.json();

            if (!res.ok) {
                throw new Error(data.error || "Failed to process image");
            }

            setResponse(data);
            setAnnotatedImage(`data:image/png;base64,${data.annotated_image}`);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="p-4">
      <h1 className="text-xl font-bold mb-4">Facial Analysis for {patient_name}</h1>
      <button
        onClick={handleCapture}
        disabled={loading}
        className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700"
      >
        {loading ? "Capturing..." : "Capture & Analyze"}
      </button>

      {error && <p className="text-red-500 mt-4">Error: {error}</p>}

      {response && (
        <div className="mt-6">
          <h2 className="font-semibold">Result:</h2>
          <p>
            <strong>Significant Change:</strong>{" "}
            {response.significant_change ? "Yes" : "No"}
          </p>
          <p>
            <strong>Change Value:</strong> {response.change_value.toFixed(4)}
          </p>
          {annotatedImage && (
            <>
            <img
              src={annotatedImage}
              alt="Annotated Face"
              className="mt-4 max-w-md border rounded"
            />

            <button onClick={onComplete}>Continue</button>
            </>
          )}
        </div>
      )}
    </div>
    );
}

export default FacialMapping;