"use client";

import { useEffect } from "react";

const SpeechEmotionAnalysis = ({patient_name, onComplete, emotion, speech}) => {
        useEffect(() => {
            const timer = setTimeout(()=> {
                console.log(`Facial mapping done for ${patient_name}`);
                onComplete();
            }, 3000);
    
            return () => clearTimeout(timer);
        }, []);
    
        return (
            <>
                <p>Running Analysis for {patient_name}...</p>
                <p>Emotion: {emotion}</p>
                <p>Speech: {speech}</p>
                <button onClick={onComplete}>Continue</button>
            </>
        );
}

export default SpeechEmotionAnalysis;