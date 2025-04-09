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
  const [response, setResponse] = useState(null);
  const [error, setError] = useState("");
  const [analyzing, setAnalyzing] = useState(false);
  const webcamRef = useRef(null);
  let intervalRef = useRef(null);
  const router = useRouter();

  
  const searchParams = useSearchParams();
  
  const patient_name = searchParams.get('patient-name');
  
  const modules = {
    emotionDetection: searchParams.get('emotion-detection') === 'true'? 1:0,
    facialMapping: searchParams.get('facial-mapping') === 'true'? 1:0,
    motionAnalysis: searchParams.get('motion-analysis') === 'true'? 1:0,
    speechAnalysis: searchParams.get('speech-analysis') === 'true'? 1:0,
  };

  const selectedModules = [];

  if (modules.motionAnalysis) selectedModules.push("motion");
  if (modules.facialMapping) selectedModules.push("facial");
  if (modules.emotionDetection) selectedModules.push("emotion");
  if (modules.speechAnalysis) selectedModules.push("speech");

  const [currentModuleIndex, setCurrentModuleIndex ] = useState(0);
  const currentModule = selectedModules[currentModuleIndex];

  const handleNextModule = () => {
    if (currentModuleIndex < selectedModules.length - 1) {
      setCurrentModuleIndex((prev) => prev + 1);
    } else {
      setAnalyzing(false);
    }
  }

  const renderCurrentModule = () => {
    switch (currentModule) {
      case "motion":
        return <MotionAnalysis patient_name={patient_name} onComplete={handleNextModule} />
      case "facial":
        return <FacialMapping patient_name={patient_name} onComplete={handleNextModule} />
      case "emotion":
      case "speech":
        return <SpeechEmotionAnalysis patient_name={patient_name} onComplete={handleNextModule} emotion={modules.emotionDetection} speech={modules.speechAnalysis}/>
      default:
        return <ReportGeneration patient_name={patient_name} />
    }
  }
  
  const startAnalysis = () => {
    setAnalyzing(true);
    setError("");
    setResponse(null);
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
      <h2 className="mb-3">Analysis for {patient_name}</h2>
  
      {!analyzing && (
        <button className="btn btn-primary" onClick={startAnalysis}>
          Start Analysis
        </button>
      )}
  
      {analyzing && renderCurrentModule()}
    </div>
  );
};

export default withAuth(AnalysisPage);
