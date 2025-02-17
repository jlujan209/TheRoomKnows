'use client';
import { useState, useEffect } from 'react';
import { useRouter, useParams } from 'next/navigation';
import withAuth from '../../hoc/withAuth';

const api_key = process.env.NEXT_PUBLIC_API_KEY;

function EditPage() {
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
            !String(patientData?.patient_age).trim() ||
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
            router.push('/home');
        } catch (error) {
            console.error("Error updating patient:", error);
            setError(error.message);
        } finally {
            setIsSubmitting(false);
        }
    };

    if (loading) return (
        <div className="container mt-4">
            <h1 className="text-center mb-4">Edit Patient Information</h1>
            <hr />
            <div className="d-flex justify-content-center align-items-center" style={{ height: '200px' }}>
                <div className="spinner-border text-primary" role="status">
                    <span className="visually-hidden">Loading...</span>
                </div>
            </div>
        </div>
    );

    if (error) return <h1>Error: {error}</h1>;

    return (
        <>  
            <div className="container mt-4">
                <h1 className="text-center mb-4">Edit Patient Information</h1>
                <hr />
                <form onSubmit={handleSubmit}>
                    <div className="mb-3">
                        <label htmlFor="patient_first_name" className="form-label">First Name</label>
                        <input
                            type="text"
                            className="form-control"
                            id="patient_first_name"
                            name="patient_first_name"
                            value={patientData?.patient_first_name || ""}
                            onChange={handleChange}
                            required
                        />
                    </div>
                    <div className="mb-3">
                        <label htmlFor="patient_last_name" className="form-label">Last Name</label>
                        <input
                            type="text"
                            className="form-control"
                            id="patient_last_name"
                            name="patient_last_name"
                            value={patientData?.patient_last_name || ""}
                            onChange={handleChange}
                            required
                        />
                    </div>
                    <div className="mb-3">
                        <label htmlFor="patient_age" className="form-label">Age</label>
                        <input
                            type="number"
                            className="form-control"
                            id="patient_age"
                            name="patient_age"
                            value={patientData?.patient_age || ""}
                            onChange={handleChange}
                            required
                        />
                    </div>
                    <div className="mb-3">
                        <label htmlFor="last_visit" className="form-label">Last Visit</label>
                        <input
                            type="date"
                            className="form-control"
                            id="last_visit"
                            name="last_visit"
                            value={patientData?.last_visit || ""}
                            onChange={handleChange}
                            required
                        />
                    </div>

                    {error && <p className="text-danger">{error}</p>}

                    <div className="d-flex justify-content-between">
                        <button type="button" className="btn btn-secondary" onClick={() => router.back()}>Cancel</button>
                        <button type="submit" className="btn btn-primary" disabled={isSubmitting}>
                            {isSubmitting ? "Updating..." : "Update Patient"}
                        </button>
                    </div>
                </form>
            </div>
        </>
    );
}

export default withAuth(EditPage);