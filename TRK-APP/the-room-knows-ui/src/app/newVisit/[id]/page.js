'use client';
import { useState, useEffect } from 'react';
import { useParams, useRouter } from 'next/navigation';

const api_key = process.env.NEXT_PUBLIC_API_KEY;

export default function NewVisitPage() {
  const { id } = useParams();
  const [patientData, setPatientData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [expanded, setExpanded] = useState({});
  const router = useRouter();

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
          setPatientData(data.patient);
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

  const toggleExpand = (option) => {
    setExpanded((prev) => ({ ...prev, [option]: !prev[option] }));
  };

  if (loading) {
    return (
      <div className="container mt-4 text-center">
        <h1 className="mb-4">Loading Patient Data...</h1>
        <div className="spinner-border text-primary" role="status">
          <span className="visually-hidden">Loading...</span>
        </div>
      </div>
    );
  }

  if (error) return <h1>Error: {error}</h1>;

  return (
    <div className="container mt-4">
      <h1 className="text-center mb-4">New Visit for {patientData?.patient_first_name} {patientData?.patient_last_name}</h1>
      <hr />
      <p className="text-center mb-4">Please select the types of analysis you wish to conduct:</p>

      <form>
        {[
          { label: "Motion Analysis", description: "Tracks body movement to detect irregular patterns." },
          { label: "Facial Mapping", description: "Analyzes facial structure for recognition and expression tracking." },
          { label: "Emotion Detection", description: "Identifies emotions based on facial expressions and voice tone." },
          { label: "Speech Analysis", description: "Evaluates speech patterns for abnormalities and coherence." }
        ].map((option, index) => (
          <div key={index} className="form-group mb-3">
            <div className="d-flex justify-content-between align-items-center">
              <label className="form-label">{option.label}</label>
              <button 
                type="button" 
                className="btn btn-link text-decoration-none"
                onClick={() => toggleExpand(option.label)}
              >
                {expanded[option.label] ? "Hide Details" : "More Info"}
              </button>
            </div>
            <div className="form-check form-switch">
              <input 
                className="form-check-input" 
                type="checkbox" 
                id={option.label.replace(/\s+/g, '')} 
                defaultChecked 
              />
              <label className="form-check-label" htmlFor={option.label.replace(/\s+/g, '')}>Enable</label>
            </div>
            {expanded[option.label] && (
              <p className="text-muted mt-2">{option.description}</p>
            )}
          </div>
        ))}

        <div className="d-flex justify-content-center gap-3 mt-4">
          <button type="button" onClick={()=> router.back()} className="btn btn-secondary">Cancel</button>
          <button type="submit" className="btn btn-primary">Start Analysis</button>
        </div>
      </form>
    </div>
  );
}
