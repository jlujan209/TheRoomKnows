"use client";

import WebcamCapture from "../../../components/WebcamCapture";
import "bootstrap/dist/css/bootstrap.min.css";
import withAuth from "../../../hoc/withAuth";

function Analysis() {
  return (
    <div className="container mt-5">
      <div className="text-center">
        <h1 style={{ color: "#0C234B", fontWeight: "bold" }}>Emotion Detection</h1>
        <p className="text-secondary" style={{ color: "#97999B" }}>
          AI-powered facial emotion analysis
        </p>
        <hr className="mb-4 border-secondary" />
      </div>

      <div className="d-flex justify-content-center">
        <WebcamCapture />
      </div>
    </div>
  );
}

export default withAuth(Analysis);