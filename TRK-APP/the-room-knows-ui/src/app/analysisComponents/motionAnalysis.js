"use client";

import { useEffect } from "react";

const MotionAnalysis = ({ patient_name, onComplete }) => {
    useEffect(() => {
        const timer = setTimeout(()=> {
            console.log(`Facial mapping done for ${patient_name}`);
            onComplete();
        }, 3000);

        return () => clearTimeout(timer);
    }, []);

    return (
            <h1>Running Facial Mapping for {patient_name}...</h1>
    );
}

export default MotionAnalysis;