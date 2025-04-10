import React, { useRef, useState, useEffect } from "react";
import Webcam from "react-webcam";
import "bootstrap/dist/css/bootstrap.min.css";
import { useRouter } from "next/navigation";
import axios from "axios";

const MotionAnalysis = ({ patient_name, onComplete }) => {
  const [recording, setRecording] = useState(false);
  const [mediaRecorder, setMediaRecorder] = useState(null);
  const [videoBlob, setVideoBlob] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [gaitResult, setGaitResult] = useState(null);
  const webcamRef = useRef(null);
  const streamRef = useRef(null);
  const videoChunksRef = useRef([]);  // Use ref to store chunks
  const router = useRouter();

  const startRecording = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    streamRef.current = stream;

    const recorder = new MediaRecorder(stream, {
      mimeType: "video/webm;codecs=vp8"
    });
    videoChunksRef.current = [];  // Reset video chunks for the new recording

    recorder.ondataavailable = (e) => {
      console.log("Recording data available, chunk size:", e.data.size);
      if (e.data.size > 0) {
        videoChunksRef.current.push(e.data);  // Use ref to accumulate chunks
      }
    };

    recorder.onstop = () => {
      console.log("Recording stopped. Final chunk count:", videoChunksRef.current.length);
      // Delay creating the final blob to ensure all chunks are accumulated
      setTimeout(() => {
        const finalBlob = new Blob([...videoChunksRef.current], { type: "video/webm" });
        console.log("Final video blob size:", finalBlob.size);
        setVideoBlob(finalBlob);
      }, 100);  // Delay to ensure all chunks are added
    };

    recorder.start();
    setMediaRecorder(recorder);
    setRecording(true);
  };

  const stopRecording = () => {
    mediaRecorder?.stop();
    streamRef.current?.getTracks().forEach((track) => track.stop());
    setRecording(false);
  };

  const uploadVideo = async () => {
    if (!videoBlob) return;
    setUploading(true);

    const formData = new FormData();
    formData.append("video", videoBlob, `${patient_name || "motion"}.webm`);
    formData.append("patient_name", patient_name || "");
    console.log("Video size:", videoBlob.size);

    try {
      const res = await axios.post("http://localhost:5000/motion-analysis/upload-video", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });

      const { abnormal_gait } = res.data;
      setGaitResult(abnormal_gait);

      const resultPayload = {
        patient_name: patient_name,
        result: abnormal_gait ? "Abnormal gait detected." : "Gait appears normal.",
      };

      await axios.post("http://localhost:5000/motion-analysis/save-results", resultPayload, {
        headers: {
          'Content-Type': 'application/json',
        }
      });

      

      onComplete();
    } catch (err) {
      console.error("Upload error:", err);
      alert("There was an error uploading the video.");
    } finally {
      setUploading(false);
    }
  };

  useEffect(() => {
    return () => {
      stopRecording();
    };
  }, []);

  return (
    <div className="container py-4 text-center">
      <h2 className="mb-4">Motion Analysis for {patient_name}</h2>

      <div className="d-flex justify-content-center mb-4">
        <Webcam
          audio={false}
          ref={webcamRef}
          mirrored
          screenshotFormat="image/png"
          className="border rounded shadow"
          style={{ width: "100%", maxWidth: "500px" }}
        />
      </div>

      <div className="d-flex justify-content-center gap-3">
        <button
          className="btn btn-secondary"
          onClick={() => router.back()}
          disabled={recording || uploading}
        >
          Cancel
        </button>

        <button
          className={`btn ${recording ? "btn-danger" : "btn-primary"}`}
          onClick={recording ? stopRecording : startRecording}
        >
          {recording ? "Stop Recording" : "Start Recording"}
        </button>

        <button
          className="btn btn-success"
          onClick={uploadVideo}
          disabled={recording || uploading || !videoBlob}
        >
          {uploading ? "Uploading..." : "Continue"}
        </button>
      </div>
    </div>
  );
};

export default MotionAnalysis;
