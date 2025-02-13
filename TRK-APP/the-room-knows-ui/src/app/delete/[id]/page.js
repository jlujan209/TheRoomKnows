'use client';
import { useParams, useRouter } from 'next/navigation'; 
import { useState, useEffect } from 'react';

const api_key = process.env.NEXT_PUBLIC_API_KEY;

export default function DeletePage() {
  const { id } = useParams();
  const router = useRouter();  // Use this to navigate after deletion
  const [firstName, setFirstName] = useState('');
  const [lastName, setLastName] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [deleting, setDeleting] = useState(false);

  if (!id) {
    return <h1>Loading...</h1>; 
  }

  // Fetch patient details
  useEffect(() => {
    const fetchPatients = async () => {
      try {
        const response = await fetch(`https://localhost:5000/patients/search?patient_id=${id}`, {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
            'API-Key': api_key
          }
        });
        if (!response.ok) {
          throw new Error(`HTTPS error! Status: ${response.status}`);
        }
        const data = await response.json();
        setFirstName(data.patient.patient_first_name);
        setLastName(data.patient.patient_last_name);    
      } catch (error) {
        setError(error.message);
      } finally {
        setLoading(false);
      }
    };

    fetchPatients();
  }, [id]);

  const handleDelete = async () => {
    setDeleting(true);
    try {
      const response = await fetch(`https://localhost:5000/patients/delete?patient_id=${id}`, {
        method: 'DELETE',
        headers: {
          'Content-Type': 'application/json',
          'API-Key': api_key
        },
        mode: 'cors'
      });
      if (!response.ok) {
        throw new Error(`Deletion failed! Status: ${response.status}`);
      }
      router.push('/'); 
    } catch (error) {
      setError(error.message);
    } finally {
      setDeleting(false);
    }
  };

  if (loading) return <p>Loading patient data...</p>;
  if (error) return <p>Error: {error}</p>;

  return (
    <>
      <h1>Deleting Patient</h1>
      <hr />
      <p>Are you sure you want to delete {firstName} {lastName}?</p>
      <div>
        <button onClick={() => router.push('/')}>Cancel</button>
        <button onClick={handleDelete} disabled={deleting}>
          {deleting ? 'Deleting...' : 'Confirm'}
        </button>
      </div>
    </>
  );
}
