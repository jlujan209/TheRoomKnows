"use client";

import React, { useRef, useState, useEffect } from "react";
import "bootstrap/dist/css/bootstrap.min.css";
import withAuth from "../../../hoc/withAuth";
import { useRouter, useSearchParams } from "next/navigation";
import FacialMapping from "../../../analysisComponents/facialMapping";
import MotionAnalysis from "../../../analysisComponents/motionAnalysis";
import SpeechEmotionAnalysis from "../../../analysisComponents/speechEmotionAnalysis";
import ReportGeneration from "../../../analysisComponents/reportGeneration";

const AnalysisPage = () => {
  const [analyzing, setAnalyzing] = useState(false);
  const [currentModuleIndex, setCurrentModuleIndex] = useState(0);
  const webcamRef = useRef(null);
  const intervalRef = useRef(null);
  const searchParams = useSearchParams();
  const router = useRouter();

  const patient_name = searchParams.get("patient-name");

  const modules = {
    emotionDetection: searchParams.get("emotion-detection") === "true",
    facialMapping: searchParams.get("facial-mapping") === "true",
    motionAnalysis: searchParams.get("motion-analysis") === "true",
    speechAnalysis: searchParams.get("speech-analysis") === "true",
  };

  const selectedModules = [
    modules.motionAnalysis && "motion",
    modules.facialMapping && "facial",
    (modules.emotionDetection || modules.speechAnalysis) && "speech-emotion",
    "report"
  ].filter(Boolean);

  const currentModule = selectedModules[currentModuleIndex];

  const handleNextModule = () => {
    if (currentModuleIndex < selectedModules.length - 1) {
      setCurrentModuleIndex((prev) => prev + 1);
    } else {
      setAnalyzing(false);
    }
  };

  const renderCurrentModule = () => {
    switch (currentModule) {
      case "motion":
        return (
          <MotionAnalysis
            patient_name={patient_name}
            onComplete={handleNextModule}
          />
        );
      case "facial":
        return (
          <FacialMapping
            patient_name={patient_name}
            onComplete={handleNextModule}
          />
        );
      case "speech-emotion":
        return (
          <SpeechEmotionAnalysis
            patient_name={patient_name}
            onComplete={handleNextModule}
            emotion={modules.emotionDetection}
            speech={modules.speechAnalysis}
          />
        );
      case "report":
        return <ReportGeneration patient_name={patient_name} motion_analysis={modules.motionAnalysis} emotion_detection={modules.emotionDetection} facial_mapping={modules.facialMapping} speech_analysis={modules.speechAnalysis}/>;
      default:
        return null;
    }
  };

  const startAnalysis = () => {
    setAnalyzing(true);
  };

  useEffect(() => {
    return () => clearInterval(intervalRef.current);
  }, []);

  return (
    <div className="container py-5">
      <div className="text-center mb-4">
        <h2 className="fw-bold">New Visit for {patient_name}</h2>
      </div>

      {!analyzing ? (
        <div className="d-flex justify-content-center">
          <button className="btn btn-lg btn-primary px-5" onClick={startAnalysis}>
            Start Visit
          </button>
        </div>
      ) : (
        <div className="analysis-module border rounded p-4 shadow-sm">
          <div className="mb-3 text-muted">
            Step {currentModuleIndex + 1} of {selectedModules.length}
          </div>
          {renderCurrentModule()}
        </div>
      )}
    </div>
  );
};

export default withAuth(AnalysisPage);
