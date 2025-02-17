'use client';
import { useState } from 'react';
import { useRouter } from 'next/navigation';
import withAuth from '../hoc/withAuth';

const api_key = process.env.NEXT_PUBLIC_API_KEY;

function AddPage() {
    return (
        <div className="container mt-5">
            <h1 className="text-center" style={{ color: "#0C234B", fontWeight: "bold" }}>Add Patient</h1>
            <p className="text-center text-secondary">Enter patient details to add them to the system</p>
            <hr className="mb-4 border-secondary" />
            <AddPatientForm />
        </div>
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

        if (Object.values(patientData).some(value => value.trim() === "")) {
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

    return (
        <form onSubmit={handleSubmit} className="mx-auto p-4 border rounded shadow-sm" style={{ maxWidth: "600px", backgroundColor: "#F4F4F4" }}>
            <div className="mb-3">
                <label className="form-label">First Name</label>
                <input 
                    type="text" 
                    name="patient_first_name"
                    value={patientData.patient_first_name} 
                    onChange={handleChange} 
                    className="form-control" 
                    required 
                />
            </div>
            <div className="mb-3">
                <label className="form-label">Last Name</label>
                <input 
                    type="text" 
                    name="patient_last_name"
                    value={patientData.patient_last_name} 
                    onChange={handleChange} 
                    className="form-control" 
                    required 
                />
            </div>
            <div className="mb-3">
                <label className="form-label">Age</label>
                <input 
                    type="number" 
                    name="patient_age"
                    value={patientData.patient_age} 
                    onChange={handleChange} 
                    className="form-control" 
                    required 
                />
            </div>
            <div className="mb-3">
                <label className="form-label">Last Visit</label>
                <input 
                    type="date" 
                    name="last_visit"
                    value={patientData.last_visit} 
                    onChange={handleChange} 
                    className="form-control" 
                    required 
                />
            </div>

            {error && <p className="text-danger">Error: {error}</p>}
            
            <div className="d-flex justify-content-between">
                <button type="button" className="btn btn-secondary" onClick={() => router.back()}>Cancel</button>
                <button type="submit" className="btn btn-primary" disabled={isSubmitting}>
                    {isSubmitting ? "Submitting..." : "Submit"}
                </button>
            </div>
        </form>
    );
}


export default withAuth(AddPage);