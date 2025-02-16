'use client';
import { useState } from 'react';
import { useRouter } from 'next/navigation';

const api_key = process.env.NEXT_PUBLIC_API_KEY;

export default function AddPage() {
    return (
        <>
            <h1>Add Patient</h1>
            <hr />
            <AddPatientForm />
        </>
    );
}

function AddPatientForm() {
    const [patientData, setPatientData] = useState({
        patient_first_name: "",
        patient_last_name: "",
        patient_age: "",
        last_visit: ""
    });

    const [error, setError] = useState(null);
    const [isSubmitting, setIsSubmitting] = useState(false);
    
    const router = useRouter();

    const handleChange = (e) => {
        const { name, value } = e.target;
        setPatientData((prevData) => ({
            ...prevData,
            [name]: value
        }));
    };

    const handleSubmit = async (e) => {
        e.preventDefault();

        if (
            !patientData.patient_first_name.trim() ||
            !patientData.patient_last_name.trim() ||
            !patientData.patient_age.trim() ||
            !patientData.last_visit.trim()
        ) {
            alert("Please fill out all required fields.");
            return;
        }

        setIsSubmitting(true);
        setError(null);

        try {
            const response = await fetch(`https://localhost:5000/patients/new`, {
                method: 'POST', 
                headers: {
                    'Content-Type': 'application/json',
                    'API-Key': api_key
                },
                body: JSON.stringify(patientData)
            });

            if (!response.ok) {
                throw new Error(`Submission failed! Status: ${response.status}`);
            }

            console.log("Patient added successfully.");
            router.push('/home'); // Redirect to home page after successful submission
        } catch (error) {
            console.error("Error adding patient:", error);
            setError(error.message);
        } finally {
            setIsSubmitting(false);
        }
    };

    const isFormValid = Object.values(patientData).every(value => value.trim() !== "");

    return (
        <form onSubmit={handleSubmit}>
            <div>
                <label>First Name:</label>
                <input 
                    type="text" 
                    name="patient_first_name"
                    value={patientData.patient_first_name} 
                    onChange={handleChange} 
                    required 
                />
            </div>
            <div>
                <label>Last Name:</label>
                <input 
                    type="text" 
                    name="patient_last_name"
                    value={patientData.patient_last_name} 
                    onChange={handleChange} 
                    required 
                />
            </div>
            <div>
                <label>Age:</label>
                <input 
                    type="text" 
                    name="patient_age"
                    value={patientData.patient_age} 
                    onChange={handleChange} 
                    required 
                />
            </div>
            <div>
                <label>Last Visit:</label>
                <input 
                    type="text" 
                    name="last_visit"
                    value={patientData.last_visit} 
                    onChange={handleChange} 
                    required 
                />
            </div>

            {error && <p style={{ color: 'red' }}>{error}</p>}
            <button type="button" onClick={()=> router.back()}>Cancel</button>
            <button type="submit" disabled={!isFormValid || isSubmitting}>
                {isSubmitting ? "Submitting..." : "Submit"}
            </button>
        </form>
    );
}
