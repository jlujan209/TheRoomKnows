'use client';
import { useParams, useRouter } from 'next/navigation'; 
import { useState, useEffect } from 'react';
import withAuth from '../../hoc/withAuth';

const api_key = process.env.NEXT_PUBLIC_API_KEY;

function DeletePage() {
  const { id } = useParams();
  const router = useRouter(); 
  const [firstName, setFirstName] = useState('');
  const [lastName, setLastName] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [deleting, setDeleting] = useState(false);

  useEffect(() => {
    const fetchPatients = async () => {
      try {
        const response = await fetch(`http://localhost:5000/patients/search?patient_id=${id}`, {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
            'API-Key': api_key
          }
        });
        if (!response.ok) {
          throw new Error(`http error! Status: ${response.status}`);
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

    if (id) {
      fetchPatients();
    }
  }, [id]);

  const handleDelete = async () => {
    setDeleting(true);
    try {
      const response = await fetch(`http://localhost:5000/patients/delete?patient_id=${id}`, {
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
      router.push('/home'); 
    } catch (error) {
      setError(error.message);
    } finally {
      setDeleting(false);
    }
  };

  if (loading) return (
    <div className="container mt-4">
      <h1 className="text-center mb-4">Delete Patient</h1>
      <hr />
      <div className="d-flex justify-content-center align-items-center" style={{ height: '200px' }}>
        <div className="spinner-border text-primary" role="status">
          <span className="visually-hidden">Loading...</span>
        </div>
      </div>
    </div>
  );

  if (error) return <p>Error: {error}</p>;

  return (
    <div className="container mt-4">
      <h1 className="text-center mb-4">Delete Patient</h1>
      <hr />
      <p className="text-center">Are you sure you want to delete <strong>{firstName} {lastName}</strong>?</p>
      <div className="d-flex justify-content-center gap-3">
        <button className="btn btn-secondary" onClick={() => router.back()}>Cancel</button>
        <button className="btn btn-danger" onClick={handleDelete} disabled={deleting}>
          {deleting ? 'Deleting...' : 'Confirm Deletion'}
        </button>
      </div>
      {error && <p className="text-danger text-center mt-3">{error}</p>}
    </div>
  );
}

export default withAuth(DeletePage);