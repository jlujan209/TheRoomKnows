"use client";


const MotionAnalysis = ({ patient_name, onComplete }) => {

    return (
        <>
            <p>Running Motion Analysis for {patient_name}...</p>
            <button onClick={onComplete}>continue</button>
        </>
    );
}

export default MotionAnalysis;