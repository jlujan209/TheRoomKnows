'use client';
import { useState, useEffect } from 'react';
import { useRouter, useParams } from 'next/navigation';

const api_key = process.env.NEXT_PUBLIC_API_KEY;

export default function EditPage() {
    const { id } = useParams();
    const router = useRouter();
    const [patientData, setPatientData] = useState({
        patient_first_name: '',
        patient_last_name: '',
        patient_age: '',
        last_visit: ''
    });
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [isSubmitting, setIsSubmitting] = useState(false);

    useEffect(() => {
        const fetchPatient = async () => {
            try {
                const response = await fetch(`https://localhost:5000/patients/search?patient_id=${id}`, {
                    method: "GET",
                    headers: {
                        "Content-Type": "application/json",
                        "API-Key": api_key
                    }
                });

                if (!response.ok) {
                    throw new Error(`Failed to fetch patient data! Status: ${response.status}`);
                }

                const data = await response.json();
                console.log(data);
                if (data) {
                    setPatientData({
                        patient_first_name: data.patient.patient_first_name || '',
                        patient_last_name: data.patient.patient_last_name || '',
                        patient_age: data.patient.patient_age || '',
                        last_visit: data.patient.last_visit || ''
                    });
                }
            } catch (error) {
                setError(error.message);
            } finally {
                setLoading(false);
            }
        };

        if (id) {
            fetchPatient();
        }
    }, [id]);

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
            !patientData?.patient_first_name?.trim() ||
            !patientData?.patient_last_name?.trim() ||
            !patientData?.patient_age?.trim() ||
            !patientData?.last_visit?.trim()
        ) {
            alert("Please fill out all required fields.");
            return;
        }

        setIsSubmitting(true);
        setError(null);

        try {
            const response = await fetch(`https://localhost:5000/patients/edit?patient_id=${id}`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                    'API-Key': api_key
                },
                body: JSON.stringify(patientData),
                mode: 'cors'
            });

            if (!response.ok) {
                throw new Error(`Update failed! Status: ${response.status}`);
            }

            console.log("Patient updated successfully.");
            router.push('/'); 
        } catch (error) {
            console.error("Error updating patient:", error);
            setError(error.message);
        } finally {
            setIsSubmitting(false);
        }
    };

    if (loading) return <h1>Loading patient data...</h1>;
    if (error) return <h1>Error: {error}</h1>;

    return (
        <>  
            <h1>Editing Patient Info</h1>
            <hr></hr>
            <form onSubmit={handleSubmit}>
                <div>
                    <label>First Name:</label>
                    <input
                        type="text"
                        name="patient_first_name"
                        value={patientData?.patient_first_name || ""}
                        onChange={handleChange}
                        required
                    />
                </div>
                <div>
                    <label>Last Name:</label>
                    <input
                        type="text"
                        name="patient_last_name"
                        value={patientData?.patient_last_name || ""}
                        onChange={handleChange}
                        required
                    />
                </div>
                <div>
                    <label>Age:</label>
                    <input
                        type="text"
                        name="patient_age"
                        value={patientData?.patient_age || ""}
                        onChange={handleChange}
                        required
                    />
                </div>
                <div>
                    <label>Last Visit:</label>
                    <input
                        type="text"
                        name="last_visit"
                        value={patientData?.last_visit || ""}
                        onChange={handleChange}
                        required
                    />
                </div>

                {error && <p style={{ color: 'red' }}>{error}</p>}
                <button type="submit" disabled={isSubmitting}>
                    {isSubmitting ? "Updating..." : "Update Patient"}
                </button>
            </form>
        </>
    );
}
